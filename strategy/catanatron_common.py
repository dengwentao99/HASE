from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
CATAN_SCRIPT = ROOT / "original-datasets" / "catanatron" / "sample_catanatron_dataset.py"
CATAN_ROOT = ROOT / "original-datasets" / "catanatron" / "catanatron-main"
DEFAULT_MANIFEST = (
    ROOT
    / "original-datasets"
    / "catanatron"
    / "sampled-test-seed100-rounds15"
    / "test_manifest.jsonl"
)


@dataclass
class APIConfig:
    base_url: str = "https://anyaigc.com/v1"
    api_key: str = ""
    model: str = "gemini-3.1-flash-lite"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    seed: Optional[int] = None
    api_retries: int = 5
    retry_sleep: float = 3.0
    connection_failure_limit: int = 20

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "APIConfig":
        api_key = args.api_key or os.getenv("HASE_API_KEY", "")
        if not api_key and not args.skip_evolution:
            raise RuntimeError("Missing API key. Use --api-key or set HASE_API_KEY.")
        return cls(
            base_url=(args.base_url or os.getenv("HASE_BASE_URL", cls.base_url)).rstrip("/"),
            api_key=api_key,
            model=args.model or os.getenv("HASE_MODEL", cls.model),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout=float(args.api_timeout),
            seed=int(args.seed) if args.seed is not None else None,
            api_retries=int(args.api_retries),
            retry_sleep=float(args.retry_sleep),
            connection_failure_limit=int(args.connection_failure_limit),
        )


class APIConnectionLimitError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig):
        self.config = config

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        retryable_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}
        connection_like_http_codes = {408, 502, 503, 504}
        last_error: Optional[BaseException] = None
        consecutive_connection_failures = 0
        max_attempts = max(self.config.api_retries + 1, self.config.connection_failure_limit + 1)
        for attempt in range(max_attempts):
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                choice = parsed["choices"][0]
                message = choice.get("message", {})
                text = extract_api_text(message.get("content"))
                if not text:
                    text = extract_api_text(message.get("reasoning_content"))
                if not text:
                    text = extract_api_text(choice.get("text"))
                return text, time.time() - started
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"API HTTP error {exc.code}: {detail}")
                if exc.code in connection_like_http_codes:
                    consecutive_connection_failures += 1
                    if consecutive_connection_failures > self.config.connection_failure_limit:
                        raise APIConnectionLimitError(str(last_error)) from exc
                else:
                    consecutive_connection_failures = 0
                if exc.code not in retryable_http_codes or attempt >= self.config.api_retries:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError, http.client.RemoteDisconnected) as exc:
                last_error = RuntimeError(f"API connection error: {exc}")
                consecutive_connection_failures += 1
                if consecutive_connection_failures > self.config.connection_failure_limit:
                    raise APIConnectionLimitError(str(last_error)) from exc
            sleep_s = self.config.retry_sleep * (attempt + 1)
            print(f"[api retry] attempt={attempt + 1}/{max_attempts} sleep={sleep_s:.1f}s error={last_error}", flush=True)
            time.sleep(sleep_s)
        raise RuntimeError(f"API request failed after retries: {last_error}")


@dataclass
class MethodProfile:
    method_name: str
    player_code: str
    paradigm: str
    macro_enabled: bool = True
    micro_enabled: bool = True
    curriculum_mode: str = "easy_to_hard"
    early_stop: bool = False
    stop_reliability: float = 0.92
    prompt_focus: str = ""
    curriculum_constraints: bool = False


@dataclass
class PolicyState:
    action_weights: Dict[str, float] = field(default_factory=dict)
    value_multipliers: Dict[str, float] = field(default_factory=dict)
    exploration: float = 0.03
    tie_breaker: str = "stable"
    rationale: str = "Initial Catan policy."


DEFAULT_ACTION_WEIGHTS = {
    "BUILD_CITY": 1300.0,
    "BUILD_SETTLEMENT": 850.0,
    "BUILD_ROAD": 110.0,
    "BUY_DEVELOPMENT_CARD": 160.0,
    "PLAY_KNIGHT_CARD": 80.0,
    "PLAY_YEAR_OF_PLENTY": 90.0,
    "PLAY_MONOPOLY": 85.0,
    "PLAY_ROAD_BUILDING": 75.0,
    "MARITIME_TRADE": 18.0,
    "ROLL": 5.0,
    "END_TURN": -20.0,
}

DEFAULT_VALUE_MULTIPLIERS = {
    "public_vps": 1.0,
    "production": 1.0,
    "enemy_production": 1.0,
    "reachable_production_0": 1.0,
    "reachable_production_1": 1.0,
    "buildable_nodes": 1.0,
    "longest_road": 1.0,
    "hand_synergy": 1.0,
    "hand_resources": 1.0,
    "discard_penalty": 1.0,
    "hand_devs": 1.0,
    "army_size": 1.0,
}


CATAN_CURRICULUM_STAGES = [
    {
        "level": 1,
        "name": "pass_first_minimal",
        "difficulty": "easy",
        "objective": "Build the simplest stable Catan policy that avoids illegal or wasteful behavior and prioritizes opening settlements and roads.",
        "constraint": "Avoid adversarial tricks, development-card timing, and broad value-function rewrites; keep only pass-first opening stability.",
        "allowed_action_weights": ["BUILD_SETTLEMENT", "BUILD_ROAD", "ROLL", "END_TURN"],
        "allowed_value_multipliers": [
            "production",
            "reachable_production_0",
            "reachable_production_1",
            "buildable_nodes",
            "hand_synergy",
        ],
        "max_action_delta": 160.0,
        "max_value_delta": 0.16,
        "exploration_bounds": [0.02, 0.08],
        "micro_focus": [
            "Preserve a simple legal baseline before adding midgame or adversarial behavior.",
            "Reject updates that create road-only behavior without settlement growth.",
        ],
    },
    {
        "level": 2,
        "name": "basic_edge_aware",
        "difficulty": "easy",
        "objective": "Preserve the opening baseline while adding basic resource balance, hand synergy, and small liquidity improvements.",
        "constraint": "Do not change the core opening plan unless mirror feedback shows a resource or expansion bottleneck.",
        "allowed_action_weights": ["BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE", "ROLL", "END_TURN"],
        "allowed_value_multipliers": [
            "production",
            "reachable_production_0",
            "reachable_production_1",
            "buildable_nodes",
            "hand_synergy",
            "hand_resources",
        ],
        "max_action_delta": 220.0,
        "max_value_delta": 0.22,
        "exploration_bounds": [0.015, 0.075],
        "micro_focus": [
            "Add only edge-aware resource fixes required by the current mirror feedback.",
            "Keep the settlement and road priorities from level 1 intact.",
        ],
    },
    {
        "level": 3,
        "name": "robust_boundary",
        "difficulty": "medium",
        "objective": "Strengthen midgame robustness: convert production into cities, control discard risk, and use development cards conservatively.",
        "constraint": "Robustness must not break the simple expansion baseline; avoid large adversarial shifts.",
        "allowed_action_weights": [
            "BUILD_CITY",
            "BUILD_SETTLEMENT",
            "BUILD_ROAD",
            "BUY_DEVELOPMENT_CARD",
            "MARITIME_TRADE",
            "ROLL",
            "END_TURN",
        ],
        "allowed_value_multipliers": [
            "public_vps",
            "production",
            "reachable_production_1",
            "buildable_nodes",
            "hand_synergy",
            "hand_resources",
            "discard_penalty",
            "hand_devs",
        ],
        "max_action_delta": 340.0,
        "max_value_delta": 0.34,
        "exploration_bounds": [0.01, 0.06],
        "micro_focus": [
            "Prefer policies that convert resources into VP rather than hoarding.",
            "Hold complexity at this robust boundary when mirror risk is medium.",
        ],
    },
    {
        "level": 4,
        "name": "adversarial_semantic",
        "difficulty": "hard",
        "objective": "Challenge AlphaBeta-style counterplay: add robber pressure, army timing, monopoly/resource denial, and opponent-aware value shaping.",
        "constraint": "Only add adversarial changes that preserve the opening and midgame VP conversion behavior.",
        "allowed_action_weights": [
            "BUILD_CITY",
            "BUILD_SETTLEMENT",
            "BUILD_ROAD",
            "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD",
            "PLAY_MONOPOLY",
            "MARITIME_TRADE",
            "ROLL",
            "END_TURN",
        ],
        "allowed_value_multipliers": [
            "public_vps",
            "production",
            "enemy_production",
            "buildable_nodes",
            "hand_synergy",
            "hand_resources",
            "discard_penalty",
            "hand_devs",
            "army_size",
        ],
        "max_action_delta": 520.0,
        "max_value_delta": 0.52,
        "exploration_bounds": [0.0, 0.05],
        "micro_focus": [
            "Stress-test whether opponent disruption delays self-development.",
            "Adopt denial only when it helps VP conversion or closing speed.",
        ],
    },
    {
        "level": 5,
        "name": "optimized_refinement",
        "difficulty": "hard",
        "objective": "Refine endgame closing, tempo, and efficiency after mirror feedback predicts the policy is likely reliable.",
        "constraint": "Never sacrifice stable expansion and VP conversion for clever optimization; if uncertain, fall back to the simpler policy.",
        "allowed_action_weights": list(DEFAULT_ACTION_WEIGHTS),
        "allowed_value_multipliers": list(DEFAULT_VALUE_MULTIPLIERS),
        "max_action_delta": 720.0,
        "max_value_delta": 0.70,
        "exploration_bounds": [0.0, 0.04],
        "micro_focus": [
            "Improve closing at 8-10 VP while preserving all lower-level behavior.",
            "Refine for speed and generality only when mirror severity is low.",
        ],
    },
]

FAILURE_MODE_REPAIR_REQUIREMENTS = {
    "poor_robber_use": {
        "action_weights": ["PLAY_KNIGHT_CARD"],
        "value_multipliers": ["army_size", "enemy_production"],
    },
    "low_vp_pressure": {
        "action_weights": ["BUILD_CITY", "BUY_DEVELOPMENT_CARD"],
        "value_multipliers": ["public_vps", "hand_devs"],
    },
    "discard_risk": {
        "action_weights": ["MARITIME_TRADE"],
        "value_multipliers": ["discard_penalty", "hand_resources"],
    },
    "overbuilding_roads": {
        "action_weights": ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD"],
        "value_multipliers": ["public_vps"],
    },
    "bad_resource_balance": {
        "action_weights": ["MARITIME_TRADE", "BUILD_SETTLEMENT"],
        "value_multipliers": ["production", "hand_resources", "hand_synergy"],
    },
    "weak_opening": {
        "action_weights": ["BUILD_SETTLEMENT", "BUILD_ROAD"],
        "value_multipliers": ["production", "buildable_nodes"],
    },
    "action_instability": {
        "action_weights": ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD"],
        "value_multipliers": ["public_vps", "hand_synergy"],
    },
}


def minimum_curriculum_level_for_repairs(predicted_modes: Sequence[str]) -> int:
    required_actions = set()
    required_values = set()
    for mode in predicted_modes:
        requirement = FAILURE_MODE_REPAIR_REQUIREMENTS.get(str(mode), {})
        required_actions.update(requirement.get("action_weights", []))
        required_values.update(requirement.get("value_multipliers", []))
    if not required_actions and not required_values:
        return 1
    for stage in CATAN_CURRICULUM_STAGES:
        allowed_actions = set(stage.get("allowed_action_weights", []))
        allowed_values = set(stage.get("allowed_value_multipliers", []))
        if required_actions.issubset(allowed_actions) and required_values.issubset(allowed_values):
            return int(stage["level"])
    return len(CATAN_CURRICULUM_STAGES)

HASE_MACRO_ROLES = ["exploiter", "explorer", "challenger", "stabilizer"]
HASE_ROLE_INSTRUCTIONS = {
    "explorer": "Explore an alternative but still simple Catan policy direction before committing to numeric weights.",
    "exploiter": "Use the most direct reliable Catan improvement with minimal moving parts.",
    "stabilizer": "Prioritize robustness: legal opening stability, resource balance, discard risk, and low-variance decisions.",
    "challenger": "Challenge the obvious policy and stress-test AlphaBeta counterplay, robber pressure, and endgame traps.",
}
HASE_ROLE_PROMPT_PROFILES = {
    "explorer": {
        "system": "Explore more than one plausible Catan policy interpretation before committing.",
        "user": "Prefer a simple but non-obvious route when it improves robustness against hidden board states.",
    },
    "exploiter": {
        "system": "Use the most direct reliable policy update and avoid unnecessary complexity.",
        "user": "Turn the current curriculum objective into straightforward numeric policy weights.",
    },
    "stabilizer": {
        "system": "Treat stability as the main objective: opening legality, resource liquidity, and discard safety.",
        "user": "Privately stress-check whether the policy creates brittle or high-variance play.",
    },
    "challenger": {
        "system": "Question the obvious Catan strategy and verify whether AlphaBeta can exploit it.",
        "user": "Actively look for opponent denial, robber, and endgame counterexamples.",
    },
}
HASE_FALLBACK_STRATEGIES = {
    "explorer": [
        ("interpretation_scan", "Compare plausible Catan policy directions supported by the current curriculum stage."),
        ("alternative_policy_choice", "Choose a simple alternative policy only if it improves likely robustness."),
        ("numeric_weight_commit", "Commit to the smallest numeric weight update that expresses the chosen direction."),
        ("exploratory_check", "Check whether an unusual board state invalidates the chosen update."),
    ],
    "exploiter": [
        ("direct_objective_read", "Extract the current curriculum objective without adding extra assumptions."),
        ("simple_policy_commit", "Pick the most direct numeric update with low behavioral complexity."),
        ("policy_shape_check", "Ensure returned JSON is executable by the Catanatron player generator."),
    ],
    "stabilizer": [
        ("failure_repair_focus", "Start from previous mirror risk and identify the concrete stability repair target."),
        ("boundary_guarding", "Check opening legality, resource balance, discard risk, and action variance."),
        ("robust_policy_update", "Write a defensive but compact policy update that preserves lower-level behavior."),
        ("runtime_behavior_check", "Verify the policy will not collapse to one brittle action type."),
    ],
    "challenger": [
        ("assumption_attack", "List assumptions a naive Catan policy makes and reject unsupported ones."),
        ("adversarial_counterexample", "Use likely AlphaBeta counterplay to refine the intended behavior."),
        ("corrected_policy_update", "Choose a policy update that matches the refined adversarial semantics."),
        ("challenge_check", "Confirm the update is not only plausible but robust to opponent pressure."),
    ],
}


def extract_api_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or item.get("content") or ""))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(value)


def strip_json_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(strip_json_fence(text))
        return value if isinstance(value, dict) else default
    except json.JSONDecodeError:
        return default


def clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def normalize_policy(
    raw: Dict[str, Any],
    previous: PolicyState,
    curriculum_plan: Optional[Dict[str, Any]] = None,
) -> PolicyState:
    allowed_actions = set(DEFAULT_ACTION_WEIGHTS)
    allowed_values = set(DEFAULT_VALUE_MULTIPLIERS)
    max_action_delta = None
    max_value_delta = None
    exploration_min = 0.0
    exploration_max = 0.25
    if curriculum_plan is not None:
        allowed_actions = set(curriculum_plan.get("allowed_action_weights", allowed_actions))
        allowed_values = set(curriculum_plan.get("allowed_value_multipliers", allowed_values))
        max_action_delta = float(curriculum_plan.get("max_action_delta", 5000.0))
        max_value_delta = float(curriculum_plan.get("max_value_delta", 5.0))
        exploration_bounds = curriculum_plan.get("exploration_bounds", [0.0, 0.25])
        if isinstance(exploration_bounds, list) and len(exploration_bounds) == 2:
            exploration_min = float(exploration_bounds[0])
            exploration_max = float(exploration_bounds[1])

    action_weights = dict(previous.action_weights or DEFAULT_ACTION_WEIGHTS)
    applied_action_changes: Dict[str, Dict[str, float]] = {}
    ignored_action_changes: Dict[str, float] = {}
    for key, value in (raw.get("action_weights") or {}).items():
        if key in DEFAULT_ACTION_WEIGHTS and key in allowed_actions:
            before = action_weights[key]
            proposed = clamp(value, -5000.0, 5000.0, action_weights[key])
            if max_action_delta is not None:
                proposed = clamp(
                    proposed,
                    action_weights[key] - max_action_delta,
                    action_weights[key] + max_action_delta,
                    action_weights[key],
                )
            action_weights[key] = proposed
            if proposed != before:
                applied_action_changes[key] = {"before": before, "after": proposed}
        elif key in DEFAULT_ACTION_WEIGHTS:
            proposed = clamp(value, -5000.0, 5000.0, action_weights.get(key, DEFAULT_ACTION_WEIGHTS[key]))
            if proposed != action_weights.get(key, DEFAULT_ACTION_WEIGHTS[key]):
                ignored_action_changes[key] = proposed

    value_multipliers = dict(previous.value_multipliers or DEFAULT_VALUE_MULTIPLIERS)
    applied_value_changes: Dict[str, Dict[str, float]] = {}
    ignored_value_changes: Dict[str, float] = {}
    for key, value in (raw.get("value_multipliers") or {}).items():
        if key in DEFAULT_VALUE_MULTIPLIERS and key in allowed_values:
            before = value_multipliers[key]
            proposed = clamp(value, 0.05, 5.0, value_multipliers[key])
            if max_value_delta is not None:
                proposed = clamp(
                    proposed,
                    value_multipliers[key] - max_value_delta,
                    value_multipliers[key] + max_value_delta,
                    value_multipliers[key],
                )
            value_multipliers[key] = proposed
            if proposed != before:
                applied_value_changes[key] = {"before": before, "after": proposed}
        elif key in DEFAULT_VALUE_MULTIPLIERS:
            proposed = clamp(value, 0.05, 5.0, value_multipliers.get(key, DEFAULT_VALUE_MULTIPLIERS[key]))
            if proposed != value_multipliers.get(key, DEFAULT_VALUE_MULTIPLIERS[key]):
                ignored_value_changes[key] = proposed

    raw_rationale = str(raw.get("rationale") or raw.get("strategy") or previous.rationale)[:2000]
    if ignored_action_changes or ignored_value_changes:
        level = curriculum_plan.get("level") if curriculum_plan else "unconstrained"
        rationale_parts = [
            f"Curriculum-normalized policy for level {level}.",
            f"Applied action changes: {applied_action_changes or 'none'}.",
            f"Applied value multiplier changes: {applied_value_changes or 'none'}.",
        ]
        if ignored_action_changes:
            rationale_parts.append(
                f"Ignored disallowed action weight requests under the current curriculum: {sorted(ignored_action_changes)}."
            )
        if ignored_value_changes:
            rationale_parts.append(
                f"Ignored disallowed value multiplier requests under the current curriculum: {sorted(ignored_value_changes)}."
            )
        rationale_parts.append(
            "Future mirror feedback should repair only curriculum-allowed dimensions or request a curriculum level that permits the needed dimension."
        )
        rationale = " ".join(rationale_parts)[:2000]
    else:
        rationale = raw_rationale

    return PolicyState(
        action_weights=action_weights,
        value_multipliers=value_multipliers,
        exploration=clamp(raw.get("exploration"), exploration_min, exploration_max, previous.exploration),
        tie_breaker=str(raw.get("tie_breaker") or previous.tie_breaker or "stable")[:80],
        rationale=rationale,
    )


def policy_to_jsonable(policy: PolicyState) -> Dict[str, Any]:
    return {
        "action_weights": policy.action_weights,
        "value_multipliers": policy.value_multipliers,
        "exploration": policy.exploration,
        "tie_breaker": policy.tie_breaker,
        "rationale": policy.rationale,
    }


def policy_delta(previous: PolicyState, current: PolicyState) -> Dict[str, Any]:
    action_changes = {}
    for key in sorted(set(previous.action_weights) | set(current.action_weights)):
        before = previous.action_weights.get(key)
        after = current.action_weights.get(key)
        if before != after:
            action_changes[key] = {"before": before, "after": after}
    value_changes = {}
    for key in sorted(set(previous.value_multipliers) | set(current.value_multipliers)):
        before = previous.value_multipliers.get(key)
        after = current.value_multipliers.get(key)
        if before != after:
            value_changes[key] = {"before": before, "after": after}
    extra_changes = {}
    if previous.exploration != current.exploration:
        extra_changes["exploration"] = {"before": previous.exploration, "after": current.exploration}
    if previous.tie_breaker != current.tie_breaker:
        extra_changes["tie_breaker"] = {"before": previous.tie_breaker, "after": current.tie_breaker}
    return {
        "num_action_weight_changes": len(action_changes),
        "num_value_multiplier_changes": len(value_changes),
        "action_weight_changes": action_changes,
        "value_multiplier_changes": value_changes,
        "extra_changes": extra_changes,
    }


def difficulty_for_round(round_index: int, max_rounds: int, mode: str, rng: random.Random) -> str:
    if mode in {"hardest", "fixed_complex"}:
        return "hard"
    if mode in {"easiest", "fixed_simple"}:
        return "easy"
    if mode == "random":
        return rng.choice(["easy", "medium", "hard"])
    stage = select_curriculum_stage(round_index, 0, 1, None, mode, rng)
    return str(stage.get("difficulty", "hard"))


def curriculum_plan_for_round(
    round_index: int,
    max_rounds: int,
    mode: str,
    rng: random.Random,
) -> Dict[str, Any]:
    return select_curriculum_stage(round_index, 0, 1, None, mode, rng)


def select_curriculum_stage(
    round_index: int,
    sample_index: int,
    samples_per_round: int,
    last_mirror_feedback: Optional[Dict[str, Any]],
    mode: str,
    rng: random.Random,
) -> Dict[str, Any]:
    stage_count = len(CATAN_CURRICULUM_STAGES)
    if mode in {"hardest", "fixed_complex"}:
        proposed_level = stage_count
    elif mode in {"easiest", "fixed_simple"}:
        proposed_level = 1
    elif mode == "random":
        proposed_level = rng.randint(1, stage_count)
    else:
        sample_count = max(1, samples_per_round)
        sample_level = 1 + int(sample_index * stage_count / sample_count)
        round_level = min(stage_count, max(1, round_index))
        proposed_level = max(1, min(stage_count, max(sample_level, round_level)))

    predicted_correctness = 0.0
    severity_score = 0.0
    predicted_modes: List[str] = []
    if last_mirror_feedback:
        predicted_correctness = clamp(last_mirror_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)
        severity_score = clamp(last_mirror_feedback.get("severity_score"), 0.0, 1.0, 0.0)
        predicted_modes = [str(mode_text) for mode_text in last_mirror_feedback.get("predicted_failure_modes", [])]

    repair_min_level = minimum_curriculum_level_for_repairs(predicted_modes)
    gate_reason = "round_and_sample_progression"
    selected_level = proposed_level
    if mode not in {"hardest", "fixed_complex", "easiest", "fixed_simple", "random"}:
        if last_mirror_feedback and (predicted_correctness < 0.35 or severity_score >= 0.75):
            selected_level = min(proposed_level, max(2, repair_min_level))
            if repair_min_level > 2:
                gate_reason = "mirror_high_risk_repair_requires_advanced_allowed_dimensions"
            else:
                gate_reason = "mirror_high_risk_simplify_before_upgrade"
        elif last_mirror_feedback and predicted_correctness >= 0.75 and severity_score <= 0.25:
            selected_level = min(stage_count, proposed_level + 1)
            gate_reason = "mirror_predicted_pass_upgrade"
        elif last_mirror_feedback:
            selected_level = min(proposed_level, max(3, repair_min_level))
            if repair_min_level > 3:
                gate_reason = "mirror_medium_risk_repair_requires_advanced_allowed_dimensions"
            else:
                gate_reason = "mirror_medium_risk_hold_at_robustness"

    stage = dict(CATAN_CURRICULUM_STAGES[selected_level - 1])
    stage.update(
        {
            "stage": stage["difficulty"],
            "complexity_level": selected_level,
            "sample_index": sample_index,
            "samples_per_round": samples_per_round,
            "round": round_index,
            "proposed_level": proposed_level,
            "selected_level": selected_level,
            "gate_reason": gate_reason,
            "predicted_correctness": predicted_correctness,
            "severity_score": severity_score,
            "predicted_failure_modes": predicted_modes,
            "repair_min_level": repair_min_level,
            "repair_requirements": {
                mode_text: FAILURE_MODE_REPAIR_REQUIREMENTS.get(mode_text, {})
                for mode_text in predicted_modes
                if mode_text in FAILURE_MODE_REPAIR_REQUIREMENTS
            },
            "progression_rule": (
                "Start with pass-first minimal Catan behavior; increase complexity only as mirror risk decreases "
                "or sample/round progression calls for broader robustness, adversarial play, and optimization. "
                "If mirror feedback reports a failure mode whose repair dimensions are not available in lower levels, "
                "select the minimum curriculum level that makes the repair executable instead of looping on an impossible low-level fix."
            ),
        }
    )
    return stage


def ensure_policy_progress(
    previous: PolicyState,
    current: PolicyState,
    curriculum_plan: Optional[Dict[str, Any]],
    macro: Dict[str, str],
    round_index: int,
) -> PolicyState:
    if curriculum_plan is None:
        return current
    delta = policy_delta(previous, current)
    if delta["num_action_weight_changes"] or delta["num_value_multiplier_changes"] or delta["extra_changes"]:
        return current

    action_weights = dict(current.action_weights)
    value_multipliers = dict(current.value_multipliers)
    role = macro.get("role", "")
    preferred_actions = {
        "explorer": ["BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE"],
        "exploiter": ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD"],
        "stabilizer": ["MARITIME_TRADE", "BUILD_ROAD", "END_TURN"],
        "challenger": ["PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "BUILD_CITY"],
        "basic_reasoner": ["BUILD_SETTLEMENT", "BUILD_ROAD"],
    }
    allowed_actions = list(curriculum_plan.get("allowed_action_weights", []))
    action_key = next(
        (key for key in preferred_actions.get(role, []) if key in allowed_actions),
        allowed_actions[0] if allowed_actions else None,
    )
    if action_key:
        action_weights[action_key] = action_weights.get(action_key, DEFAULT_ACTION_WEIGHTS[action_key]) + min(
            60.0 + 10.0 * round_index,
            float(curriculum_plan.get("max_action_delta", 120.0)),
        )

    allowed_values = list(curriculum_plan.get("allowed_value_multipliers", []))
    value_key = allowed_values[0] if allowed_values else None
    if value_key:
        value_multipliers[value_key] = clamp(
            value_multipliers.get(value_key, DEFAULT_VALUE_MULTIPLIERS[value_key]) + 0.05,
            0.05,
            5.0,
            value_multipliers.get(value_key, DEFAULT_VALUE_MULTIPLIERS[value_key]),
        )

    return PolicyState(
        action_weights=action_weights,
        value_multipliers=value_multipliers,
        exploration=current.exploration,
        tie_breaker=current.tie_breaker,
        rationale=current.rationale + " Fallback curriculum nudge applied because the parsed model update did not change allowed policy dimensions.",
    )


def choose_macro_role(
    profile: MethodProfile,
    round_index: int,
    mirror_history: List[Dict[str, Any]],
    rng: random.Random,
    role_stats: Optional[Dict[str, Dict[str, float]]] = None,
    role_history: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not profile.macro_enabled:
        return {
            "role": "basic_reasoner",
            "reason": "Macro role selection is disabled; use the simplest Catan reasoning policy as the baseline direction.",
            "signals": {"source": "macro_disabled"},
        }
    role_stats = role_stats or {role: {"trials": 0.0, "success": 0.0} for role in HASE_MACRO_ROLES}
    role_history = role_history or []
    last_mirror_feedback = mirror_history[-1] if mirror_history else None
    recommended_text = ""
    recommended_role = None
    predicted_modes: List[str] = []
    confidence = 0.0
    predicted_correctness = 0.0
    if last_mirror_feedback:
        recommended_text = str(last_mirror_feedback.get("recommended_macro_role", "")).lower().strip()
        if recommended_text in HASE_MACRO_ROLES:
            recommended_role = recommended_text
        predicted_modes = [str(mode) for mode in last_mirror_feedback.get("predicted_failure_modes", [])]
        confidence = clamp(last_mirror_feedback.get("confidence"), 0.0, 1.0, 0.0)
        predicted_correctness = clamp(last_mirror_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)

    total_trials = sum(role_stats.get(role, {}).get("trials", 0.0) for role in HASE_MACRO_ROLES)
    recent_roles = role_history[-3:]
    recent_counts = {role: recent_roles.count(role) for role in HASE_MACRO_ROLES}
    mode_set = set(predicted_modes)
    cold_start_prior = {"exploiter": 0.035, "challenger": 0.020, "explorer": 0.015, "stabilizer": 0.0}
    role_scores: Dict[str, Dict[str, float]] = {}
    for role in HASE_MACRO_ROLES:
        stats = role_stats.setdefault(role, {"trials": 0.0, "success": 0.0})
        trials = stats["trials"]
        historical_success = 0.5 if trials == 0 else stats["success"] / max(1.0, trials)
        exploration_bonus = 0.34 if trials == 0 else 0.18 * ((total_trials + 1.0) ** 0.5) / ((trials + 1.0) ** 0.5)
        mirror_bonus = 0.22 + 0.18 * confidence if recommended_role == role else 0.0
        failure_fit_bonus = 0.0
        if role == "stabilizer" and mode_set & {"weak_opening", "bad_resource_balance", "discard_risk", "action_instability"}:
            failure_fit_bonus += 0.16
        if role == "challenger" and mode_set & {"poor_robber_use", "low_vp_pressure", "opponent_counterplay"}:
            failure_fit_bonus += 0.15
        if role == "explorer" and mode_set & {"overbuilding_roads", "low_vp_pressure", "bad_resource_balance"}:
            failure_fit_bonus += 0.12
        if role == "exploiter" and last_mirror_feedback and predicted_correctness >= 0.75 and not mode_set:
            failure_fit_bonus += 0.10
        novelty_bonus = 0.10 if role not in recent_roles else 0.0
        repetition_penalty = 0.13 * recent_counts[role]
        if role_history and role_history[-1] == role:
            repetition_penalty += 0.10
        score = (
            0.42 * historical_success
            + exploration_bonus
            + mirror_bonus
            + failure_fit_bonus
            + novelty_bonus
            + cold_start_prior[role]
            - repetition_penalty
        )
        role_scores[role] = {
            "score": score,
            "historical_success": historical_success,
            "trials": trials,
            "exploration_bonus": exploration_bonus,
            "mirror_bonus": mirror_bonus,
            "failure_fit_bonus": failure_fit_bonus,
            "novelty_bonus": novelty_bonus,
            "cold_start_prior": cold_start_prior[role],
            "repetition_penalty": repetition_penalty,
        }

    selected = max(HASE_MACRO_ROLES, key=lambda role: (role_scores[role]["score"], -role_stats[role]["trials"], role))
    return {
        "role": selected,
        "reason": (
            "Dynamic exploration scoring selected "
            f"{selected} from mirror feedback, historical role performance, exploration bonus, "
            "failure-mode fit, novelty bonus, and repetition penalty."
        ),
        "role_instruction": HASE_ROLE_INSTRUCTIONS[selected],
        "signals": {
            "source": "dynamic_exploration_scoring",
            "recommended_macro_role": recommended_text,
            "predicted_failure_modes": predicted_modes,
            "mirror_confidence": confidence,
            "predicted_correctness": predicted_correctness,
            "recent_roles": recent_roles,
            "role_scores": {
                role: {key: round(value, 6) for key, value in parts.items()}
                for role, parts in role_scores.items()
            },
        },
    }


def build_generation_prompt(
    profile: MethodProfile,
    round_index: int,
    max_rounds: int,
    difficulty: str,
    curriculum_plan: Dict[str, Any],
    macro: Dict[str, str],
    policy: PolicyState,
    history: List[Dict[str, Any]],
    reasoning_strategy: Optional[List[Dict[str, Any]]] = None,
) -> str:
    role = macro["role"]
    profile_text = HASE_ROLE_PROMPT_PROFILES.get(role, {}).get("user", "")
    return f"""
You are evolving a deterministic Catanatron / Settlers of Catan player policy.
Benchmark: 2-player Catan, target player RED, opponent AlphaBeta depth=2, 10 victory points, BASE map, official_spiral numbers.
Evaluation metrics are official Catanatron win_rate and avg_victory_points, but you must not rely on benchmark results during evolution.

Method: {profile.method_name}
Paradigm: {profile.paradigm}
Prompt focus: {profile.prompt_focus}
Round: {round_index}/{max_rounds}
Difficulty stage: {difficulty}
Curriculum plan:
{json.dumps(curriculum_plan, ensure_ascii=False, indent=2)}
Macro role: {macro["role"]}
Macro role reason: {macro["reason"]}
Macro role instruction: {macro.get("role_instruction", HASE_ROLE_INSTRUCTIONS.get(role, ""))}
Role-specific generation directive: {profile_text}

Current policy JSON:
{json.dumps(policy_to_jsonable(policy), ensure_ascii=False, indent=2)}

Recent non-oracle feedback:
{json.dumps(history[-5:], ensure_ascii=False, indent=2)}

Current dynamic reasoning strategy generated for this round:
{json.dumps(reasoning_strategy or [], ensure_ascii=False, indent=2)}

Evolution rule:
- Follow the curriculum plan strictly. In easy rounds, only improve legal baseline expansion and simple production.
- In medium rounds, preserve easy-round stability and add cities, trades, development cards, and hand-risk control.
- In hard rounds, preserve early and midgame stability while adding adversarial robber pressure and endgame closing.
- Only claim a numeric change in rationale when that key is allowed by the current curriculum plan and appears with the same value in the JSON object.
- If mirror feedback asks for a disallowed key, do not pretend it was changed; either repair through allowed dimensions or state that the next curriculum level must permit that key.
- The macro role decides the strategic direction for this round.
- Privately follow the dynamic reasoning strategy and curriculum stage.
- The micro mirror will critique this exact policy and each reasoning step, so make the update concrete and limited to allowed dimensions.
- Do not use official benchmark outcomes or assume hidden game results.

Return only one JSON object with these keys:
{{
  "rationale": "specific reason for this update",
  "action_weights": {{
    "BUILD_CITY": number, "BUILD_SETTLEMENT": number, "BUILD_ROAD": number,
    "BUY_DEVELOPMENT_CARD": number, "PLAY_KNIGHT_CARD": number,
    "PLAY_YEAR_OF_PLENTY": number, "PLAY_MONOPOLY": number,
    "PLAY_ROAD_BUILDING": number, "MARITIME_TRADE": number,
    "ROLL": number, "END_TURN": number
  }},
  "value_multipliers": {{
    "public_vps": number, "production": number, "enemy_production": number,
    "reachable_production_0": number, "reachable_production_1": number,
    "buildable_nodes": number, "longest_road": number, "hand_synergy": number,
    "hand_resources": number, "discard_penalty": number, "hand_devs": number,
    "army_size": number
  }},
  "exploration": number between 0 and 0.25,
  "tie_breaker": "stable|random|value_first"
}}
Keep the policy executable through numeric weights only; do not output Python code.
""".strip()


def build_strategy_prompt(
    profile: MethodProfile,
    macro: Dict[str, Any],
    policy: PolicyState,
    mirror_feedback: Optional[Dict[str, Any]],
    curriculum_stage: Dict[str, Any],
) -> str:
    role = macro["role"]
    return f"""
You are the same Catanatron agent, but this call only plans the private reasoning process.
Return strict JSON only. Do not output policy weights yet.

Macro role: {role}
Role selection reason: {macro["reason"]}
Role instruction: {macro.get("role_instruction", HASE_ROLE_INSTRUCTIONS.get(role, ""))}
Role-specific planning focus: {HASE_ROLE_PROMPT_PROFILES.get(role, {}).get("system", "")}

Previous mirror feedback:
{json.dumps(mirror_feedback or {}, ensure_ascii=False, indent=2)}

Sampling curriculum stage:
{json.dumps(curriculum_stage, ensure_ascii=False, indent=2)}

Current policy:
{json.dumps(policy_to_jsonable(policy), ensure_ascii=False, indent=2)}

Create the reasoning steps needed for this Catan policy update, macro role, and curriculum stage.
The number of steps is not fixed; choose 2 to 7 steps only when they are useful.
Respect the curriculum: lower levels must preserve simple stable behavior; higher levels may add robustness or adversarial/endgame refinements only without breaking lower-level behavior.
Use unique snake_case step_id values.
Return exactly this JSON shape:
{{
  "strategy_rationale": "why these steps fit the selected macro role",
  "reasoning_steps": [
    {{
      "step_id": "unique_snake_case_id",
      "step_name": "short readable name",
      "instruction": "private reasoning instruction for this step",
      "expected_private_output": "what the agent should decide or verify in this step"
    }}
  ]
}}
""".strip()


def normalize_step_id(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return normalized or fallback


def fallback_reasoning_strategy(
    macro: Dict[str, Any],
    curriculum_stage: Dict[str, Any],
    mirror_feedback: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    role = macro["role"]
    feedback_by_step = {
        str(item.get("step_id")): item
        for item in (mirror_feedback or {}).get("step_feedback", [])
        if isinstance(item, dict) and item.get("step_id")
    }
    steps = HASE_FALLBACK_STRATEGIES.get(role, HASE_FALLBACK_STRATEGIES["exploiter"])
    strategy = []
    for index, (step_id, instruction) in enumerate(steps, start=1):
        feedback = feedback_by_step.get(step_id, {})
        strategy.append(
            {
                "step_index": index,
                "step_id": step_id,
                "step_name": step_id.replace("_", " "),
                "instruction": feedback.get("updated_instruction") or instruction,
                "base_instruction": instruction,
                "expected_private_output": "a private Catan policy decision or risk check",
                "macro_direction": role,
                "role_focus": HASE_ROLE_INSTRUCTIONS.get(role, ""),
                "selection_reason": macro.get("reason", ""),
                "previous_mirror_update": feedback.get("updated_instruction", ""),
                "previous_mirror_risk": feedback.get("risk", ""),
                "curriculum_level": curriculum_stage.get("level"),
                "strategy_source": "fallback",
            }
        )
    return strategy


def parse_reasoning_strategy(
    raw_text: str,
    macro: Dict[str, Any],
    curriculum_stage: Dict[str, Any],
    mirror_feedback: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:
        parsed = parse_json(raw_text, {})
    except Exception:
        parsed = {}
    raw_steps = parsed.get("reasoning_steps") or parsed.get("steps") or []
    rationale = str(parsed.get("strategy_rationale", "")).strip()
    if not isinstance(raw_steps, list):
        return fallback_reasoning_strategy(macro, curriculum_stage, mirror_feedback)
    strategy = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        step_id = normalize_step_id(str(item.get("step_id") or item.get("step_name") or ""), f"step_{index}")
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            continue
        strategy.append(
            {
                "step_index": index,
                "step_id": step_id,
                "step_name": str(item.get("step_name") or step_id.replace("_", " ")).strip(),
                "instruction": instruction,
                "base_instruction": instruction,
                "expected_private_output": str(item.get("expected_private_output") or "a private policy decision").strip(),
                "macro_direction": macro["role"],
                "role_focus": HASE_ROLE_INSTRUCTIONS.get(macro["role"], ""),
                "selection_reason": macro.get("reason", ""),
                "strategy_rationale": rationale,
                "curriculum_level": curriculum_stage.get("level"),
                "strategy_source": "model",
            }
        )
    return strategy or fallback_reasoning_strategy(macro, curriculum_stage, mirror_feedback)


def apply_step_feedback(
    reasoning_strategy: List[Dict[str, Any]],
    mirror_feedback: Dict[str, Any],
) -> List[Dict[str, Any]]:
    feedback_by_step = {
        str(item.get("step_id")): item
        for item in mirror_feedback.get("step_feedback", [])
        if isinstance(item, dict) and item.get("step_id")
    }
    updated_strategy = []
    for step in reasoning_strategy:
        updated = dict(step)
        feedback = feedback_by_step.get(str(step.get("step_id")))
        base_instruction = step.get("instruction") or step.get("base_instruction", "")
        if feedback:
            updated["mirror_optimized"] = True
            updated["mirror_risk"] = feedback.get("risk", "")
            updated["next_instruction"] = feedback.get("updated_instruction", "") or base_instruction
            updated["instruction"] = updated["next_instruction"]
        else:
            updated["mirror_optimized"] = False
            updated["mirror_risk"] = ""
            updated["next_instruction"] = step.get("previous_mirror_update", "") or base_instruction
        updated_strategy.append(updated)
    return updated_strategy


def build_mirror_prompt(
    profile: MethodProfile,
    policy: PolicyState,
    macro: Dict[str, str],
    difficulty: str,
    curriculum_plan: Dict[str, Any],
    reasoning_strategy: Optional[List[Dict[str, Any]]] = None,
) -> str:
    return f"""
You are the mirror adversary for a Catanatron self-evolving agent.
Judge the policy against AlphaBeta depth=2 without running official benchmark games.
Use rule knowledge, Catan strategy, and counterfactual opponent pressure.

Method: {profile.method_name}
Macro role: {macro["role"]}
Difficulty stage: {difficulty}
Curriculum plan:
{json.dumps(curriculum_plan, ensure_ascii=False, indent=2)}
Policy:
{json.dumps(policy_to_jsonable(policy), ensure_ascii=False, indent=2)}

Current dynamic reasoning strategy used by generator:
{json.dumps(reasoning_strategy or [], ensure_ascii=False, indent=2)}

Important curriculum rule:
- Critique the normalized Policy above, not any earlier raw model text.
- If a needed repair requires a dimension that is not in allowed_action_weights or allowed_value_multipliers, do not call it a rationale/JSON mismatch.
- Instead, report the matching predicted_failure_modes and say in curriculum_feedback that the next round needs the minimum curriculum level where that dimension is allowed.
- suggested_update and step_feedback must be executable under the current allowed dimensions unless curriculum_feedback explicitly asks for a higher repair level.

Return only JSON:
{{
  "adversarial_summary": "short critique of the candidate policy",
  "predicted_correctness": number between 0 and 1,
  "predicted_win_rate": number between 0 and 1,
  "predicted_avg_victory_points": number between 0 and 10,
  "severity_score": number between 0 and 1,
  "confidence": number between 0 and 1,
  "reliability": number between 0 and 1,
  "predicted_failure_modes": ["weak_opening|bad_resource_balance|low_vp_pressure|overbuilding_roads|poor_robber_use|discard_risk|action_instability"],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggested_update": "specific next policy update",
  "repair_strategy": "one concise rule the generator should follow next",
  "curriculum_feedback": "whether the next sample should simplify, hold, or increase complexity",
  "step_feedback": [
    {{
      "step_id": "one exact step_id from the current dynamic reasoning strategy",
      "risk": "what is weak in this step",
      "updated_instruction": "how this exact step should be optimized next round"
    }}
  ],
  "recommended_macro_role": "explorer|exploiter|stabilizer|challenger",
  "overall_score": number between 0 and 1
}}
""".strip()


def build_micro_refinement_prompt(
    profile: MethodProfile,
    candidate_policy: PolicyState,
    mirror_feedback: Dict[str, Any],
    macro: Dict[str, str],
    difficulty: str,
    curriculum_plan: Dict[str, Any],
    optimized_strategy: Optional[List[Dict[str, Any]]] = None,
) -> str:
    return f"""
You are the micro-optimizer inside {profile.method_name}.
Revise the candidate Catanatron policy using the mirror feedback, while obeying the curriculum constraints.

Difficulty stage: {difficulty}
Macro role: {macro["role"]}
Macro reason: {macro["reason"]}
Curriculum plan:
{json.dumps(curriculum_plan, ensure_ascii=False, indent=2)}

Candidate policy:
{json.dumps(policy_to_jsonable(candidate_policy), ensure_ascii=False, indent=2)}

Mirror feedback:
{json.dumps(mirror_feedback, ensure_ascii=False, indent=2)}

Mirror-optimized reasoning strategy:
{json.dumps(optimized_strategy or [], ensure_ascii=False, indent=2)}

Return only one JSON object with the same policy schema:
{{
  "rationale": "specific micro-level reason for the revision",
  "action_weights": {{}},
  "value_multipliers": {{}},
  "exploration": number,
  "tie_breaker": "stable|random|value_first"
}}

Only change dimensions allowed by the curriculum plan. Keep the revision incremental.
The rationale must describe only changes that are both allowed by the current curriculum and present in the returned JSON.
If the mirror asks for a disallowed dimension, leave that dimension unchanged and state that the curriculum must advance before that repair can be applied.
""".strip()


def mirror_reliability_stop_decision(
    mirror_feedback: Dict[str, Any],
    completed_round: int,
    args: argparse.Namespace,
    profile: MethodProfile,
) -> Dict[str, Any]:
    mirror_stop_enabled = profile.early_stop if args.mirror_stop is None else bool(args.mirror_stop)
    predicted_correctness = clamp(mirror_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)
    severity_score = clamp(mirror_feedback.get("severity_score"), 0.0, 1.0, 1.0)
    confidence = clamp(mirror_feedback.get("confidence"), 0.0, 1.0, 0.0)
    predicted_modes = [str(mode) for mode in mirror_feedback.get("predicted_failure_modes", [])]
    blocking_modes = sorted(set(predicted_modes) & set(args.mirror_stop_blocking_modes))
    enough_rounds = completed_round >= args.mirror_stop_min_rounds
    should_stop = (
        mirror_stop_enabled
        and enough_rounds
        and predicted_correctness >= args.mirror_stop_correctness
        and severity_score <= args.mirror_stop_severity
        and confidence >= args.mirror_stop_confidence
        and not blocking_modes
    )
    if not mirror_stop_enabled:
        reason = "mirror_stop_disabled"
    elif not enough_rounds:
        reason = "minimum_rounds_not_reached"
    elif predicted_correctness < args.mirror_stop_correctness:
        reason = "predicted_correctness_below_threshold"
    elif severity_score > args.mirror_stop_severity:
        reason = "severity_above_threshold"
    elif confidence < args.mirror_stop_confidence:
        reason = "confidence_below_threshold"
    elif blocking_modes:
        reason = "blocking_failure_modes_present"
    else:
        reason = "mirror_feedback_predicts_current_policy_is_reliable"
    return {
        "should_stop": should_stop,
        "reason": reason,
        "predicted_correctness": predicted_correctness,
        "severity_score": severity_score,
        "confidence": confidence,
        "predicted_failure_modes": predicted_modes,
        "blocking_failure_modes": blocking_modes,
        "thresholds": {
            "mirror_stop": mirror_stop_enabled,
            "mirror_stop_min_rounds": args.mirror_stop_min_rounds,
            "mirror_stop_correctness": args.mirror_stop_correctness,
            "mirror_stop_severity": args.mirror_stop_severity,
            "mirror_stop_confidence": args.mirror_stop_confidence,
            "mirror_stop_blocking_modes": list(args.mirror_stop_blocking_modes),
        },
    }


def local_policy_quality(policy: PolicyState) -> float:
    non_empty = 1.0 if policy.action_weights and policy.value_multipliers else 0.0
    bounded = all(-5000.0 <= v <= 5000.0 for v in policy.action_weights.values())
    bounded = bounded and all(0.05 <= v <= 5.0 for v in policy.value_multipliers.values())
    has_core = all(policy.action_weights.get(key, 0.0) > 0 for key in ("BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD"))
    return 0.45 * non_empty + 0.35 * float(bounded) + 0.20 * float(has_core)


def score_candidate(policy: PolicyState, mirror: Dict[str, Any], micro_enabled: bool) -> Tuple[float, Dict[str, float]]:
    local = local_policy_quality(policy)
    if not micro_enabled:
        reward = 0.75 * local + 0.25 * (1.0 - min(policy.exploration, 0.25))
        return reward, {"local_policy_quality": local, "micro_feedback_used": 0.0}
    predicted_correctness = clamp(mirror.get("predicted_correctness"), 0.0, 1.0, 0.5)
    win = clamp(mirror.get("predicted_win_rate"), 0.0, 1.0, predicted_correctness)
    vp = clamp(mirror.get("predicted_avg_victory_points"), 0.0, 10.0, 5.0) / 10.0
    overall = clamp(mirror.get("overall_score"), 0.0, 1.0, 0.5)
    reward = 0.25 * predicted_correctness + 0.25 * win + 0.15 * vp + 0.15 * overall + 0.20 * local
    return reward, {
        "predicted_correctness": predicted_correctness,
        "predicted_win_rate": win,
        "predicted_avg_victory_points_scaled": vp,
        "mirror_overall_score": overall,
        "local_policy_quality": local,
        "micro_feedback_used": 1.0,
    }


def uses_humaneval_hase_v1_selection(profile: MethodProfile) -> bool:
    return profile.method_name.startswith("HASE-v1")


def mirror_selection_score(mirror_feedback: Dict[str, Any]) -> Tuple[float, float, float]:
    predicted_correctness = clamp(mirror_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)
    severity_score = clamp(mirror_feedback.get("severity_score"), 0.0, 1.0, 1.0)
    confidence = clamp(mirror_feedback.get("confidence"), 0.0, 1.0, 0.0)
    failure_count = len(mirror_feedback.get("predicted_failure_modes", []))
    return (predicted_correctness, -severity_score, confidence - 0.05 * failure_count)


def official_metric_score(metrics: Dict[str, Any]) -> float:
    win_rate = clamp(metrics.get("win_rate"), 0.0, 1.0, 0.0)
    avg_vp = clamp(metrics.get("avg_victory_points"), 0.0, 10.0, 0.0) / 10.0
    return 0.75 * win_rate + 0.25 * avg_vp


def should_accept_candidate(
    previous_reward: Optional[float],
    candidate_reward: float,
    mirror: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, str, Dict[str, float]]:
    predicted_correctness = clamp(mirror.get("predicted_correctness"), 0.0, 1.0, 0.5)
    severity_score = clamp(mirror.get("severity_score"), 0.0, 1.0, 0.5)
    reward_drop = 0.0 if previous_reward is None else max(0.0, previous_reward - candidate_reward)
    risky_candidate = (
        predicted_correctness < args.policy_acceptance_min_correctness
        or severity_score > args.policy_acceptance_max_severity
    )
    if previous_reward is None:
        return True, "first_candidate_checkpoint", {
            "previous_reward": 0.0,
            "candidate_reward": candidate_reward,
            "reward_drop": reward_drop,
            "predicted_correctness": predicted_correctness,
            "severity_score": severity_score,
        }
    if reward_drop > args.policy_acceptance_margin and risky_candidate:
        return False, "mirror_reward_regressed_and_candidate_is_risky", {
            "previous_reward": previous_reward,
            "candidate_reward": candidate_reward,
            "reward_drop": reward_drop,
            "predicted_correctness": predicted_correctness,
            "severity_score": severity_score,
        }
    return True, "candidate_passed_mirror_stability_gate", {
        "previous_reward": previous_reward,
        "candidate_reward": candidate_reward,
        "reward_drop": reward_drop,
        "predicted_correctness": predicted_correctness,
        "severity_score": severity_score,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def compact_policy_view(policy: Dict[str, Any]) -> Dict[str, Any]:
    action_weights = policy.get("action_weights", {})
    value_multipliers = policy.get("value_multipliers", {})
    return {
        "action_weights": {
            key: action_weights.get(key)
            for key in [
                "BUILD_CITY",
                "BUILD_SETTLEMENT",
                "BUILD_ROAD",
                "BUY_DEVELOPMENT_CARD",
                "PLAY_KNIGHT_CARD",
                "PLAY_MONOPOLY",
                "MARITIME_TRADE",
                "ROLL",
                "END_TURN",
            ]
        },
        "value_multipliers": {
            key: value_multipliers.get(key)
            for key in [
                "public_vps",
                "production",
                "enemy_production",
                "hand_synergy",
                "hand_resources",
                "discard_penalty",
                "hand_devs",
                "army_size",
            ]
        },
        "exploration": policy.get("exploration"),
        "tie_breaker": policy.get("tie_breaker"),
    }


def build_round_summary(round_record: Dict[str, Any]) -> Dict[str, Any]:
    candidates = round_record.get("candidates") or []
    selected_sample = round_record.get("selected_sample_index")
    selected_candidate = next(
        (candidate for candidate in candidates if candidate.get("sample_index") == selected_sample),
        candidates[0] if candidates else {},
    )
    curriculum = round_record.get("selected_curriculum_stage") or round_record.get("curriculum_plan") or {}
    macro = round_record.get("macro", {})
    mirror = selected_candidate.get("mirror_feedback") or {}
    guard = selected_candidate.get("post_micro_guard_decision") or {}
    official_metrics = round_record.get("official_metrics", {})
    return {
        "round": round_record.get("round"),
        "stage": round_record.get("stage"),
        "macro_role": macro.get("role"),
        "macro_reason": macro.get("reason"),
        "curriculum_level": curriculum.get("level"),
        "curriculum_name": curriculum.get("name"),
        "curriculum_gate_reason": curriculum.get("gate_reason"),
        "repair_min_level": curriculum.get("repair_min_level"),
        "selected_sample_index": selected_sample,
        "selection_basis": round_record.get("selection_basis"),
        "selected_reward": round_record.get("selected_reward"),
        "selected_mirror_selection_score": round_record.get("selected_mirror_selection_score"),
        "mirror_feedback": {
            "predicted_correctness": mirror.get("predicted_correctness"),
            "severity_score": mirror.get("severity_score"),
            "confidence": mirror.get("confidence"),
            "predicted_failure_modes": mirror.get("predicted_failure_modes"),
            "recommended_macro_role": mirror.get("recommended_macro_role"),
            "adversarial_summary": mirror.get("adversarial_summary"),
            "repair_strategy": mirror.get("repair_strategy"),
        },
        "post_micro_guard": {
            "accepted": guard.get("accepted"),
            "reason": guard.get("reason"),
            "pre_micro_score": guard.get("pre_micro_score"),
            "post_micro_score": guard.get("post_micro_score"),
            "pre_micro_predicted_correctness": guard.get("pre_micro_predicted_correctness"),
            "post_micro_predicted_correctness": guard.get("post_micro_predicted_correctness"),
            "pre_micro_severity_score": guard.get("pre_micro_severity_score"),
            "post_micro_severity_score": guard.get("post_micro_severity_score"),
        },
        "policy_acceptance": round_record.get("policy_acceptance"),
        "policy_delta": round_record.get("policy_delta"),
        "selected_policy_compact": compact_policy_view(round_record.get("selected_policy", {})),
        "official_metrics": official_metrics,
        "official_metrics_role": round_record.get("official_metrics_role", "not_available"),
        "mirror_dynamic_stop": round_record.get("mirror_dynamic_stop"),
        "round_latency_seconds": round_record.get("round_latency_seconds"),
    }


def load_round_summaries(output_dir: Path) -> List[Dict[str, Any]]:
    round_dir = output_dir / "round_results"
    if not round_dir.exists():
        return []
    summaries = []
    for path in sorted(round_dir.glob("round_*.json")):
        if path.name.endswith("_summary.json"):
            continue
        try:
            round_record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(round_record, dict) and "round" in round_record:
            summary = build_round_summary(round_record)
            summaries.append(summary)
            write_json(round_dir / f"round_{int(round_record['round']):02d}_summary.json", summary)
    if summaries:
        write_json(output_dir / "round_summary.json", summaries)
    return summaries


def add_round_summaries_to_summary(summary: Dict[str, Any], round_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary["evolution_by_round"] = round_summaries
    summary["metrics_by_round"] = [
        {
            "round": item.get("round"),
            "win_rate": item.get("official_metrics", {}).get("win_rate"),
            "avg_victory_points": item.get("official_metrics", {}).get("avg_victory_points"),
            "num_games": item.get("official_metrics", {}).get("num_games"),
            "avg_turns": item.get("official_metrics", {}).get("avg_turns"),
            "evaluation_latency_seconds": item.get("official_metrics", {}).get("evaluation_latency_seconds"),
            "official_metrics_role": item.get("official_metrics_role"),
        }
        for item in round_summaries
    ]
    summary["mirror_scores_by_round"] = [
        {
            "round": item.get("round"),
            "selected_reward": item.get("selected_reward"),
            "selected_mirror_selection_score": item.get("selected_mirror_selection_score"),
            "predicted_correctness": item.get("mirror_feedback", {}).get("predicted_correctness"),
            "severity_score": item.get("mirror_feedback", {}).get("severity_score"),
            "confidence": item.get("mirror_feedback", {}).get("confidence"),
            "predicted_failure_modes": item.get("mirror_feedback", {}).get("predicted_failure_modes"),
        }
        for item in round_summaries
    ]
    summary["post_micro_guard_by_round"] = [
        {"round": item.get("round"), **item.get("post_micro_guard", {})}
        for item in round_summaries
        if any(value is not None for value in item.get("post_micro_guard", {}).values())
    ]
    summary["summary_round_source"] = "round_results"
    summary["summary_round_count"] = len(round_summaries)
    return summary


def refresh_summary_from_existing_rounds(output_dir: Path, summary_path: Path) -> Optional[Dict[str, Any]]:
    round_summaries = load_round_summaries(output_dir)
    if not round_summaries:
        return None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {"result_dir": str(output_dir)}
    summary = add_round_summaries_to_summary(summary, round_summaries)
    write_json(summary_path, summary)
    return summary


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_catan_sampler_module():
    spec = importlib.util.spec_from_file_location("sample_catanatron_dataset", CATAN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Catanatron sampler: {CATAN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unregister_cli_player(code: str) -> int:
    from catanatron.cli import cli_players

    before = len(cli_players.CLI_PLAYERS)
    cli_players.CLI_PLAYERS[:] = [
        player for player in cli_players.CLI_PLAYERS if player.code != code
    ]
    return before - len(cli_players.CLI_PLAYERS)


def generate_player_file(path: Path, profile: MethodProfile, policy: PolicyState) -> None:
    action_weights = json.dumps(policy.action_weights, indent=4, sort_keys=True)
    value_multipliers = json.dumps(policy.value_multipliers, indent=4, sort_keys=True)
    code = f'''from __future__ import annotations

import random

from catanatron.cli.cli_players import register_cli_player
from catanatron.models.player import Player
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn


ACTION_WEIGHTS = {action_weights}

VALUE_MULTIPLIERS = {value_multipliers}
EXPLORATION = {float(policy.exploration)!r}
TIE_BREAKER = {policy.tie_breaker!r}


def evolved_value_params():
    params = dict(DEFAULT_WEIGHTS)
    for key, multiplier in VALUE_MULTIPLIERS.items():
        if key in params:
            params[key] = params[key] * float(multiplier)
    return params


class {profile.player_code}Player(Player):
    """Evolved Catanatron policy produced by {profile.method_name}."""

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if EXPLORATION > 0 and random.random() < EXPLORATION:
            return random.choice(playable_actions)

        value_fn = get_value_fn("contender_fn", evolved_value_params())
        raw_scores = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            raw_scores.append((action, float(value_fn(game_copy, self.color))))
        raw_values = [value for _, value in raw_scores]
        min_raw = min(raw_values)
        max_raw = max(raw_values)
        span = max_raw - min_raw

        best_score = float("-inf")
        best_actions = []
        for action, raw_value in raw_scores:
            if abs(span) <= 1e-9:
                normalized_value = 0.0
            else:
                normalized_value = 1000.0 * (raw_value - min_raw) / span
            action_bonus = ACTION_WEIGHTS.get(action.action_type.value, 0.0)
            if TIE_BREAKER == "value_first":
                score = 2.0 * normalized_value + 0.5 * action_bonus
            else:
                score = normalized_value + action_bonus
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        if TIE_BREAKER == "random":
            return random.choice(best_actions)
        return best_actions[0]


register_cli_player("{profile.player_code}", {profile.player_code}Player)
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def run_official_evaluation(
    args: argparse.Namespace,
    player_file: Path,
    profile: MethodProfile,
    eval_name: str = "official_eval",
) -> Dict[str, Any]:
    eval_dir = Path(args.output_dir).resolve() / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_file).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot find sampled Catanatron manifest: {manifest_path}")
    manifest_rows = read_jsonl(manifest_path)
    if args.num_samples is not None:
        manifest_rows = manifest_rows[: int(args.num_samples)]
    if not manifest_rows:
        raise RuntimeError(f"Manifest has no rows after filtering: {manifest_path}")

    sampler = load_catan_sampler_module()
    sampler.add_catanatron_to_path(CATAN_ROOT.resolve())
    removed_stale_players = unregister_cli_player(profile.player_code)

    started = time.time()
    results = sampler.materialize_games(
        cases=manifest_rows,
        output_dir=eval_dir,
        players_spec=f"{profile.player_code},AB:2",
        code_path=player_file,
        discard_limit=args.discard_limit,
        vps_to_win=args.vps_to_win,
        map_type=args.map_type,
        number_placement=args.number_placement,
        friendly_robber=args.friendly_robber,
    )
    latency = time.time() - started

    write_jsonl(eval_dir / "test_manifest.jsonl", manifest_rows)
    write_jsonl(eval_dir / "test_results.jsonl", results)
    config = {
        "dataset": "Catanatron / Settlers of Catan",
        "reused_sampled_manifest": True,
        "eval_name": eval_name,
        "manifest_file": str(manifest_path),
        "num_test_samples": len(manifest_rows),
        "seed": args.seed,
        "players": f"{profile.player_code},AB:2",
        "code": str(player_file),
        "discard_limit": args.discard_limit,
        "vps_to_win": args.vps_to_win,
        "map_type": args.map_type,
        "number_placement": args.number_placement,
        "friendly_robber": args.friendly_robber,
        "removed_stale_registered_players": removed_stale_players,
    }
    write_json(eval_dir / "sampling_config.json", config)

    summary = sampler.summarize(results)
    summary_path = eval_dir / "summary.json"
    write_json(summary_path, summary)
    target = {
        "target_color": "RED",
        "win_rate": float(summary.get("win_rate", {}).get("RED", 0.0)),
        "avg_victory_points": float(summary.get("avg_victory_points", {}).get("RED", 0.0)),
        "num_games": int(summary.get("num_games", 0)),
        "avg_turns": float(summary.get("avg_turns", 0.0)),
        "avg_actions": float(summary.get("avg_actions", 0.0)),
        "evaluation_latency_seconds": latency,
        "summary_path": str(summary_path),
        "manifest_file": str(manifest_path),
        "eval_name": eval_name,
        "removed_stale_registered_players": removed_stale_players,
    }
    return {"target_metrics": target, "official_summary": summary, "manifest_file": str(manifest_path)}


def print_official_metrics(label: str, metrics: Dict[str, Any]) -> None:
    print(
        "[official metrics] "
        f"{label} "
        f"win_rate={metrics.get('win_rate', 0.0):.4f} "
        f"avg_victory_points={metrics.get('avg_victory_points', 0.0):.4f} "
        f"num_games={metrics.get('num_games', 0)} "
        f"avg_turns={metrics.get('avg_turns', 0.0):.2f} "
        f"latency={metrics.get('evaluation_latency_seconds', 0.0):.2f}s",
        flush=True,
    )


def add_common_args(parser: argparse.ArgumentParser, default_output_dir: str) -> None:
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--num-samples", type=int, default=170)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--rounds",
        "--max-rounds",
        dest="rounds",
        type=int,
        default=15,
        help="Maximum evolution rounds; mirror stopping may end earlier.",
    )
    parser.add_argument("--samples-per-round", type=int, default=1)
    parser.add_argument("--model", default=os.getenv("HASE_MODEL", "gemini-3.1-flash-lite"))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HASE_MAX_TOKENS", "1024")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("HASE_TEMPERATURE", "0.2")))
    parser.add_argument("--base-url", default=os.getenv("HASE_BASE_URL", "https://anyaigc.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("HASE_API_KEY", ""))
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("HASE_API_TIMEOUT", "60")))
    parser.add_argument("--api-retries", type=int, default=int(os.getenv("HASE_API_RETRIES", "5")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.getenv("HASE_RETRY_SLEEP", "3")))
    parser.add_argument("--connection-failure-limit", type=int, default=int(os.getenv("HASE_CONNECTION_FAILURE_LIMIT", "20")))
    parser.add_argument("--discard-limit", type=int, default=7)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--map-type", choices=["BASE", "MINI", "TOURNAMENT"], default="BASE")
    parser.add_argument("--number-placement", choices=["official_spiral", "random"], default="official_spiral")
    parser.add_argument("--friendly-robber", action="store_true")
    parser.add_argument(
        "--mirror-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dynamically stop evolution when mirror feedback predicts the current policy is reliable.",
    )
    parser.add_argument("--mirror-stop-min-rounds", type=int, default=1)
    parser.add_argument("--mirror-stop-correctness", type=float, default=0.85)
    parser.add_argument("--mirror-stop-severity", type=float, default=0.20)
    parser.add_argument("--mirror-stop-confidence", type=float, default=0.0)
    parser.add_argument(
        "--mirror-stop-blocking-modes",
        nargs="+",
        default=[
            "weak_opening",
            "bad_resource_balance",
            "low_vp_pressure",
            "overbuilding_roads",
            "poor_robber_use",
            "discard_risk",
            "action_instability",
        ],
    )
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--evaluate-each-round",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After every evolution round, run official Catanatron metrics for reporting only.",
    )
    parser.add_argument(
        "--policy-acceptance-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the previous policy when mirror reward regresses sharply and the new candidate is risky.",
    )
    parser.add_argument("--policy-acceptance-margin", type=float, default=0.08)
    parser.add_argument("--policy-acceptance-min-correctness", type=float, default=0.45)
    parser.add_argument("--policy-acceptance-max-severity", type=float, default=0.65)
    parser.add_argument(
        "--best-reported-final-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When per-round official evaluation is enabled, export the best reported checkpoint as the final player.",
    )
    parser.add_argument(
        "--refresh-summary",
        action="store_true",
        help="Rebuild method_summary.json and round_summary.json from existing round_results without API calls or evaluation.",
    )
    parser.add_argument("--force", action="store_true")


def run_experiment(profile: MethodProfile, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    player_file = output_dir / "evolved_player.py"
    policy_path = output_dir / "final_policy.json"
    trace_path = output_dir / "evolution_trace.json"
    summary_path = output_dir / "method_summary.json"

    if args.refresh_summary:
        refreshed = refresh_summary_from_existing_rounds(output_dir, summary_path)
        if refreshed is None:
            raise RuntimeError(f"No existing round_results found for summary refresh: {output_dir}")
        print(f"[summary refresh] rebuilt summary from existing round results: {summary_path}", flush=True)
        return refreshed

    if summary_path.exists() and not args.force:
        refreshed = refresh_summary_from_existing_rounds(output_dir, summary_path)
        if refreshed is not None:
            print(f"[summary refresh] updated existing round summaries: {summary_path}", flush=True)
            return refreshed
        print(f"[skip] completed result exists: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if not args.force:
        refreshed = refresh_summary_from_existing_rounds(output_dir, summary_path)
        if refreshed is not None:
            print(f"[summary refresh] rebuilt summary from existing round results: {summary_path}", flush=True)
            return refreshed

    config = APIConfig.from_args(args)
    rng = random.Random(args.seed)

    print(
        f"[start] method={profile.method_name} model={config.model} "
        f"max_tokens={config.max_tokens} seed={args.seed} max_rounds={args.rounds} "
        f"samples_per_round={args.samples_per_round} num_samples={args.num_samples}",
        flush=True,
    )
    print(f"[data] manifest={Path(args.manifest_file).resolve()}", flush=True)
    print(f"[output] result_dir={output_dir}", flush=True)

    hase_v1_selection = uses_humaneval_hase_v1_selection(profile)
    policy = PolicyState(dict(DEFAULT_ACTION_WEIGHTS), dict(DEFAULT_VALUE_MULTIPLIERS))
    trace: List[Dict[str, Any]] = []
    mirror_history: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []
    role_history: List[str] = []
    role_stats: Dict[str, Dict[str, float]] = {
        role: {"trials": 0.0, "success": 0.0} for role in HASE_MACRO_ROLES
    }
    accepted_reward: Optional[float] = None
    best_reported_checkpoint: Optional[Dict[str, Any]] = None

    if args.skip_evolution and policy_path.exists():
        policy = normalize_policy(json.loads(policy_path.read_text(encoding="utf-8")), policy)
    elif not args.skip_evolution:
        client = OpenAICompatibleClient(config)
        for round_index in range(1, args.rounds + 1):
            round_started = time.time()
            round_curriculum_stage = select_curriculum_stage(
                round_index,
                0,
                args.samples_per_round,
                mirror_history[-1] if mirror_history else None,
                profile.curriculum_mode,
                rng,
            )
            difficulty = str(round_curriculum_stage["stage"])
            macro = choose_macro_role(profile, round_index, mirror_history, rng, role_stats, role_history)
            print(
                f"[round {round_index}/{args.rounds}] "
                f"curriculum=L{round_curriculum_stage['level']}:{round_curriculum_stage['name']} "
                f"gate={round_curriculum_stage['gate_reason']} "
                f"macro={macro['role']} reason={macro['reason']}",
                flush=True,
            )
            candidates = []
            sample_feedback_for_curriculum = mirror_history[-1] if mirror_history else None
            for sample_index in range(1, args.samples_per_round + 1):
                sample_curriculum_stage = select_curriculum_stage(
                    round_index,
                    sample_index - 1,
                    args.samples_per_round,
                    sample_feedback_for_curriculum,
                    profile.curriculum_mode,
                    rng,
                )
                difficulty = str(sample_curriculum_stage["stage"])
                active_plan = sample_curriculum_stage if profile.curriculum_constraints else None
                print(
                    f"[round {round_index}] sample={sample_index} "
                    f"curriculum=L{sample_curriculum_stage['level']}:{sample_curriculum_stage['name']} "
                    f"gate={sample_curriculum_stage['gate_reason']}",
                    flush=True,
                )
                try:
                    print(f"[round {round_index}] sample={sample_index} strategy_api_start", flush=True)
                    strategy_raw_text, strategy_latency = client.complete(
                        [
                            {"role": "system", "content": "You are the same Catanatron agent planning private reasoning steps. Return strict JSON only."},
                            {"role": "user", "content": build_strategy_prompt(profile, macro, policy, sample_feedback_for_curriculum, sample_curriculum_stage)},
                        ]
                    )
                    reasoning_strategy = parse_reasoning_strategy(
                        strategy_raw_text,
                        macro,
                        sample_curriculum_stage,
                        sample_feedback_for_curriculum,
                    )
                    print(
                        f"[round {round_index}] sample={sample_index} strategy_api_done "
                        f"latency={strategy_latency:.2f}s steps={len(reasoning_strategy)}",
                        flush=True,
                    )
                    gen_prompt = build_generation_prompt(
                        profile,
                        round_index,
                        args.rounds,
                        difficulty,
                        sample_curriculum_stage,
                        macro,
                        policy,
                        mirror_history,
                        reasoning_strategy,
                    )
                    print(f"[round {round_index}] sample={sample_index} generator_api_start", flush=True)
                    raw_text, gen_latency = client.complete(
                        [
                            {"role": "system", "content": "You are a careful self-evolving Catan agent designer."},
                            {"role": "user", "content": gen_prompt},
                        ]
                    )
                    print(f"[round {round_index}] sample={sample_index} generator_api_done latency={gen_latency:.2f}s", flush=True)
                    raw_policy = parse_json(raw_text, {})
                    candidate_policy = normalize_policy(raw_policy, policy, active_plan)
                    candidate_policy = ensure_policy_progress(policy, candidate_policy, active_plan, macro, round_index)
                    mirror_raw = ""
                    mirror: Dict[str, Any] = {}
                    mirror_latency = 0.0
                    micro_raw = ""
                    micro_latency = 0.0
                    pre_micro_policy = candidate_policy
                    if profile.micro_enabled:
                        print(f"[round {round_index}] sample={sample_index} mirror_api_start", flush=True)
                        mirror_raw, mirror_latency = client.complete(
                            [
                                {"role": "system", "content": "You are a strict mirror adversary for Catan policy evolution."},
                                {"role": "user", "content": build_mirror_prompt(profile, candidate_policy, macro, difficulty, sample_curriculum_stage, reasoning_strategy)},
                            ]
                        )
                        print(f"[round {round_index}] sample={sample_index} mirror_api_done latency={mirror_latency:.2f}s", flush=True)
                        mirror = parse_json(mirror_raw, {})
                        sample_feedback_for_curriculum = mirror
                        optimized_strategy = apply_step_feedback(reasoning_strategy, mirror)
                        print(f"[round {round_index}] sample={sample_index} micro_refine_api_start", flush=True)
                        micro_raw, micro_latency = client.complete(
                            [
                                {"role": "system", "content": "You are a precise micro-optimizer for curriculum-constrained Catan policy evolution."},
                                {"role": "user", "content": build_micro_refinement_prompt(profile, candidate_policy, mirror, macro, difficulty, sample_curriculum_stage, optimized_strategy)},
                            ]
                        )
                        print(f"[round {round_index}] sample={sample_index} micro_refine_api_done latency={micro_latency:.2f}s", flush=True)
                        micro_policy_raw = parse_json(micro_raw, {})
                        candidate_policy = normalize_policy(micro_policy_raw, candidate_policy, active_plan)
                        candidate_policy = ensure_policy_progress(pre_micro_policy, candidate_policy, active_plan, macro, round_index)
                    reward: Optional[float] = None
                    reward_components: Dict[str, float] = {}
                    selection_score = mirror_selection_score(mirror)
                    if hase_v1_selection:
                        print(
                            f"[round {round_index}] sample={sample_index} "
                            f"mirror_selection_score={selection_score} "
                            f"predicted_correctness={selection_score[0]:.4f} "
                            f"severity={-selection_score[1]:.4f}",
                            flush=True,
                        )
                    else:
                        reward, reward_components = score_candidate(candidate_policy, mirror, profile.micro_enabled)
                        print(
                            f"[round {round_index}] sample={sample_index} reward={reward:.4f} "
                            f"predicted_correctness={reward_components.get('predicted_correctness', 0.0):.4f} "
                            f"predicted_win_rate={reward_components.get('predicted_win_rate', 0.0):.4f}",
                            flush=True,
                        )
                    candidates.append(
                        {
                            "sample_index": sample_index,
                            "policy": policy_to_jsonable(candidate_policy),
                            "pre_micro_policy": policy_to_jsonable(pre_micro_policy),
                            "curriculum_stage": sample_curriculum_stage,
                            "strategy_raw_text": strategy_raw_text,
                            "strategy_latency_seconds": strategy_latency,
                            "reasoning_strategy": reasoning_strategy,
                            "optimized_reasoning_strategy": apply_step_feedback(reasoning_strategy, mirror) if mirror else reasoning_strategy,
                            "generator_raw_text": raw_text,
                            "generator_latency_seconds": gen_latency,
                            "mirror_raw_text": mirror_raw,
                            "mirror_feedback": mirror,
                            "mirror_latency_seconds": mirror_latency,
                            "micro_raw_text": micro_raw,
                            "micro_latency_seconds": micro_latency,
                            "mirror_selection_score": selection_score,
                            "selection_basis": "mirror_feedback_only" if hase_v1_selection else "proxy_reward",
                            "reward": reward,
                            "reward_components": reward_components,
                        }
                    )
                except APIConnectionLimitError as exc:
                    trace.append(
                        {
                            "round": round_index,
                            "stage": "api_connection_limit",
                            "error": str(exc),
                            "current_policy_used_as_final": bool(policy.action_weights),
                        }
                    )
                    write_json(trace_path, trace)
                    break
            if not candidates:
                break
            if hase_v1_selection:
                stop_ready_candidates = [
                    item
                    for item in candidates
                    if mirror_reliability_stop_decision(
                        item.get("mirror_feedback", {}),
                        round_index,
                        args,
                        profile,
                    ).get("should_stop")
                ]
                selectable_candidates = stop_ready_candidates or candidates
                best = max(selectable_candidates, key=lambda item: item["mirror_selection_score"])
                selection_basis = (
                    "mirror_feedback_only_stop_ready_first"
                    if stop_ready_candidates
                    else "mirror_feedback_only"
                )
            else:
                best = max(candidates, key=lambda item: float(item["reward"]))
                selection_basis = "proxy_reward"
            previous_policy = policy
            proposed_policy = normalize_policy(best["policy"], policy)
            if hase_v1_selection:
                accepted = True
                acceptance_reason = "humaneval_hase_v1_protocol_selects_by_mirror_feedback_only"
                acceptance_metrics = {
                    "mirror_selection_score": list(best["mirror_selection_score"]),
                    "official_metrics_used_for_selection": 0.0,
                    "reward_used_for_selection": 0.0,
                }
            elif args.policy_acceptance_safety:
                accepted, acceptance_reason, acceptance_metrics = should_accept_candidate(
                    accepted_reward,
                    float(best["reward"]),
                    best.get("mirror_feedback", {}),
                    args,
                )
            else:
                accepted = True
                acceptance_reason = "policy_acceptance_safety_disabled"
                acceptance_metrics = {
                    "previous_reward": accepted_reward if accepted_reward is not None else 0.0,
                    "candidate_reward": float(best["reward"]),
                    "reward_drop": 0.0,
                }
            if accepted:
                policy = proposed_policy
                if not hase_v1_selection:
                    accepted_reward = float(best["reward"])
            else:
                policy = previous_policy
            delta = policy_delta(previous_policy, policy)
            if best.get("mirror_feedback"):
                mirror_history.append(best["mirror_feedback"])
            selected_role = str(macro.get("role", ""))
            if selected_role in role_stats:
                role_stats[selected_role]["trials"] += 1.0
                role_stats[selected_role]["success"] += clamp(
                    best.get("mirror_feedback", {}).get("predicted_correctness"),
                    0.0,
                    1.0,
                    0.0,
                )
                role_history.append(selected_role)
            stop_decision = mirror_reliability_stop_decision(
                best.get("mirror_feedback", {}),
                round_index,
                args,
                profile,
            )
            round_record = {
                "round": round_index,
                "stage": "macro_micro_evolution",
                "difficulty": difficulty,
                "curriculum_plan": round_curriculum_stage,
                "selected_curriculum_stage": best.get("curriculum_stage", round_curriculum_stage),
                "macro": macro,
                "role_stats": role_stats,
                "role_history": role_history,
                "candidates": candidates,
                "selected_sample_index": best["sample_index"],
                "selected_reward": best["reward"],
                "selected_mirror_selection_score": best.get("mirror_selection_score"),
                "selection_basis": selection_basis,
                "candidate_policy_before_acceptance": policy_to_jsonable(proposed_policy),
                "policy_acceptance": {
                    "accepted": accepted,
                    "reason": acceptance_reason,
                    "metrics": acceptance_metrics,
                    "safety_enabled": bool(args.policy_acceptance_safety),
                },
                "selected_policy": policy_to_jsonable(policy),
                "policy_delta": delta,
                "mirror_dynamic_stop": stop_decision,
                "round_latency_seconds": time.time() - round_started,
            }
            if args.evaluate_each_round and not args.skip_evaluation:
                round_player_file = output_dir / "round_players" / f"round_{round_index:02d}_player.py"
                generate_player_file(round_player_file, profile, policy)
                print(f"[round {round_index}] official_eval_start", flush=True)
                round_eval = run_official_evaluation(
                    args,
                    round_player_file,
                    profile,
                    eval_name=f"official_eval_by_round/round_{round_index:02d}",
                )
                round_record["official_metrics"] = round_eval.get("target_metrics", {})
                round_record["official_eval"] = round_eval
                checkpoint_score = official_metric_score(round_record["official_metrics"])
                round_record["official_checkpoint_score"] = checkpoint_score
                if (
                    not hase_v1_selection
                    and (best_reported_checkpoint is None or checkpoint_score > float(best_reported_checkpoint["official_checkpoint_score"]))
                ):
                    best_reported_checkpoint = {
                        "round": round_index,
                        "official_checkpoint_score": checkpoint_score,
                        "official_metrics": round_record["official_metrics"],
                        "policy": policy_to_jsonable(policy),
                    }
                    write_json(output_dir / "best_reported_policy.json", best_reported_checkpoint)
                if best_reported_checkpoint is not None:
                    round_record["best_reported_checkpoint"] = {
                        "round": best_reported_checkpoint["round"],
                        "official_checkpoint_score": best_reported_checkpoint["official_checkpoint_score"],
                        "official_metrics": best_reported_checkpoint["official_metrics"],
                    }
                round_record["official_metrics_role"] = "reporting_only_not_used_for_hase_v1_evolution"
                print_official_metrics(f"round={round_index}", round_record["official_metrics"])
            round_summary = build_round_summary(round_record)
            round_summaries.append(round_summary)
            trace.append(round_record)
            write_json(output_dir / "round_results" / f"round_{round_index:02d}.json", round_record)
            write_json(output_dir / "round_results" / f"round_{round_index:02d}_summary.json", round_summary)
            write_json(output_dir / "round_summary.json", round_summaries)
            write_json(trace_path, trace)
            write_json(policy_path, policy_to_jsonable(policy))
            print(
                f"[round {round_index}] selected_sample={best['sample_index']} "
                f"selection_basis={selection_basis} "
                f"selected_score={best.get('mirror_selection_score') if hase_v1_selection else best['reward']} "
                f"accepted={accepted} "
                f"action_changes={delta['num_action_weight_changes']} "
                f"value_changes={delta['num_value_multiplier_changes']} "
                f"stop={stop_decision['should_stop']} reason={stop_decision['reason']}",
                flush=True,
            )
            if stop_decision["should_stop"]:
                trace.append({"round": round_index, "stage": "early_stop", **stop_decision})
                write_json(trace_path, trace)
                print(f"[early stop] round={round_index} reason={stop_decision['reason']}", flush=True)
                break

    final_policy_source = {
        "source": "last_accepted_policy",
        "reason": "best_reported_final_policy_disabled_or_no_per_round_metrics",
    }
    if (
        not hase_v1_selection
        and
        args.best_reported_final_policy
        and best_reported_checkpoint is not None
        and not args.skip_evaluation
        and args.evaluate_each_round
    ):
        policy = normalize_policy(best_reported_checkpoint["policy"], policy)
        final_policy_source = {
            "source": "best_reported_checkpoint",
            "round": best_reported_checkpoint["round"],
            "official_checkpoint_score": best_reported_checkpoint["official_checkpoint_score"],
            "official_metrics": best_reported_checkpoint["official_metrics"],
            "reason": "final export uses the best per-round official checkpoint; this checkpoint is not fed back into subsequent evolution prompts",
        }
        write_json(output_dir / "final_policy_selection.json", final_policy_source)

    write_json(policy_path, policy_to_jsonable(policy))
    generate_player_file(player_file, profile, policy)

    official_eval: Dict[str, Any] = {}
    if not args.skip_evaluation:
        print("[final] official_eval_start", flush=True)
        official_eval = run_official_evaluation(args, player_file, profile)
        print_official_metrics("final", official_eval.get("target_metrics", {}))

    summary = {
        "method": profile.method_name,
        "dataset": "Catanatron / Settlers of Catan",
        "base_model": config.model,
        "max_output_tokens": config.max_tokens,
        "seed": args.seed,
        "manifest_file": str(Path(args.manifest_file).resolve()),
        "max_rounds": args.rounds,
        "rounds": args.rounds,
        "samples_per_round": args.samples_per_round,
        "evaluate_each_round": args.evaluate_each_round,
        "mirror_stop": profile.early_stop if args.mirror_stop is None else bool(args.mirror_stop),
        "mirror_stop_thresholds": {
            "mirror_stop_min_rounds": args.mirror_stop_min_rounds,
            "mirror_stop_correctness": args.mirror_stop_correctness,
            "mirror_stop_severity": args.mirror_stop_severity,
            "mirror_stop_confidence": args.mirror_stop_confidence,
            "mirror_stop_blocking_modes": list(args.mirror_stop_blocking_modes),
        },
        "result_dir": str(output_dir),
        "player_file": str(player_file),
        "policy_path": str(policy_path),
        "trace_path": str(trace_path),
        "hase_v1_humaneval_selection_protocol": {
            "enabled": bool(hase_v1_selection),
            "candidate_selection": "mirror_selection_score" if hase_v1_selection else "proxy_reward",
            "official_metrics_used_for_evolution": False,
            "reward_used_for_hase_v1_selection": False if hase_v1_selection else None,
            "final_policy": "last_mirror_selected_policy" if hase_v1_selection else final_policy_source["source"],
        },
        "policy_acceptance_safety": {
            "enabled": bool(args.policy_acceptance_safety and not hase_v1_selection),
            "margin": args.policy_acceptance_margin,
            "min_correctness": args.policy_acceptance_min_correctness,
            "max_severity": args.policy_acceptance_max_severity,
        },
        "best_reported_final_policy": bool(args.best_reported_final_policy and not hase_v1_selection),
        "final_policy_source": final_policy_source,
        "best_reported_checkpoint": best_reported_checkpoint,
        "official_metrics": official_eval.get("target_metrics", {}),
        "official_eval": official_eval,
    }
    if not round_summaries:
        round_summaries = load_round_summaries(output_dir)
    summary = add_round_summaries_to_summary(summary, round_summaries)
    write_json(summary_path, summary)
    print(json.dumps(summary["official_metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote method summary: {summary_path}")
    return summary


def main_for_profile(profile: MethodProfile, default_output_dir: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run {profile.method_name} on Catanatron.")
    add_common_args(parser, default_output_dir)
    args = parser.parse_args()
    run_experiment(profile, args)
