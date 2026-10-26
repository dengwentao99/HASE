from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from minerva_evaluation import (
    CATEGORY_ORDER,
    aggregate_sample_results,
    answers_equivalent,
    category_for_type,
    evaluate_solution,
    evaluator_capabilities,
    extract_final_answer,
    format_metric_line,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "original-datasets" / "minerva-math" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results-minerva" / "MinervaMath_20260719" / "HASE-v1-v2-domain-aware"

METHOD_NAME = "HASE-v1 post-micro-guard-v2 Minerva Math"
MATH_FAILURE_MODES = {
    "problem_misread",
    "algebra_error",
    "calculus_error",
    "unit_error",
    "approximation_error",
    "missing_case",
    "logical_gap",
    "answer_format_error",
}

DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "chemistry": {
        "reasoning_focus": [
            "track units and dimensions through every substitution",
            "preserve scientific notation and powers of ten",
            "check stoichiometric, thermodynamic, and material-property assumptions",
            "match requested significant figures",
        ],
        "common_risks": ["unit conversion", "power-of-ten error", "molar versus particle quantity", "significant figures"],
        "role_prior": {"stabilizer": 0.08, "exploiter": 0.05, "challenger": 0.03, "explorer": 0.0},
        "required_verification": "dimensional analysis, scientific-notation recomputation, and requested precision",
    },
    "mathematics": {
        "reasoning_focus": [
            "preserve symbolic equivalence rather than matching surface form",
            "check domains, constants of integration, initial conditions, and missing solution branches",
            "distinguish general solutions from particular solutions",
            "verify transforms, derivatives, and substitutions independently",
        ],
        "common_risks": ["missing integration constant", "lost solution branch", "invalid domain", "non-equivalent symbolic rewrite"],
        "role_prior": {"challenger": 0.08, "explorer": 0.06, "stabilizer": 0.04, "exploiter": 0.02},
        "required_verification": "symbolic substitution or differentiation, domain checks, and condition satisfaction",
    },
    "physics": {
        "reasoning_focus": [
            "identify the governing physical law before numerical substitution",
            "maintain unit-system consistency and dimensional correctness",
            "check signs, limiting behavior, approximation validity, and significant figures",
            "separate exact expressions from requested numerical approximations",
        ],
        "common_risks": ["wrong physical regime", "unit conversion", "sign error", "premature rounding"],
        "role_prior": {"stabilizer": 0.08, "challenger": 0.05, "exploiter": 0.04, "explorer": 0.01},
        "required_verification": "independent formula derivation, dimensional analysis, scale check, and precision check",
    },
    "other": {
        "reasoning_focus": [
            "state the operative definition or model assumption explicitly",
            "distinguish equilibrium, marginal, average, and elasticity quantities",
            "check signs, interpretation, and admissible ranges",
            "connect the mathematical result back to the requested concept",
        ],
        "common_risks": ["definition mismatch", "unstated economic assumption", "sign interpretation", "invalid range"],
        "role_prior": {"explorer": 0.08, "challenger": 0.05, "stabilizer": 0.03, "exploiter": 0.02},
        "required_verification": "definition check, assumption check, algebraic recomputation, and interpretation",
    },
}

MACRO_ROLES = ("explorer", "exploiter", "stabilizer", "challenger")
ROLE_INSTRUCTIONS = {
    "explorer": (
        "Compare plausible formulations as verification alternatives. Keep the inherited answer as the default; "
        "change it only after a concrete parent error is localized and an independent derivation confirms a replacement."
    ),
    "exploiter": (
        "Exploit the strongest verified structure already present. Preserve the inherited answer and verified steps "
        "unless a benchmark-local contradiction requires a narrowly justified correction."
    ),
    "stabilizer": (
        "Preserve correct intermediate relationships, repair local algebra, units, precision, or "
        "formatting, and return the inherited answer unchanged when no localized error is proved."
    ),
    "challenger": (
        "Try to falsify the inherited solution using the exact problem conventions. A failed attack strengthens the "
        "parent and must not trigger an answer change; external conventions cannot override explicit problem data."
    ),
}

STRICT_JSON_PROMPT_RULES = """
JSON output rules:
- Return one complete JSON object and no markdown fence or commentary.
- Keep the response compact enough to finish every required field; prefer concise equations to repeated prose.
- Use plain ASCII mathematics in final_answer when possible.
- Every literal backslash inside a JSON string must be JSON-escaped as two backslash characters.
- Never leave a string, brace, or JSON object unfinished.
""".strip()

MINERVA_EVIDENCE_HIERARCHY = """
Minerva evidence hierarchy:
1. The exact target question and every value, convention, approximation, and completed subproblem explicitly supplied in the prompt are authoritative.
2. A direct derivation under those supplied conventions is stronger than recalled textbook values or alternative real-world conventions.
3. External knowledge may check consistency but must not replace prompt-specific data.
4. Solve only the final requested subproblem; do not reopen completed preamble subproblems unless the target explicitly asks for it.
""".strip()

PARENT_ANSWER_CHANGE_GATE = """
Parent-preservation protocol:
- The inherited solution is a protected baseline, not merely another draft.
- Style, added detail, a different convention, role novelty, or mirror uncertainty is not evidence for changing its final answer.
- Change the inherited final answer only when all three conditions hold: a precise parent equation or assumption is shown wrong under the problem statement; an independent recomputation produces a different answer; and the replacement satisfies units, domain, sign, and requested precision.
- If any condition is missing, preserve the inherited final answer and make only local explanatory or formatting repairs.
""".strip()

ROLE_FALLBACK_STEPS = {
    "explorer": [
        ("identify_formulations", "List the viable mathematical formulations and required assumptions."),
        ("select_route", "Choose the least fragile route that can reach the requested quantity."),
        ("derive_and_check", "Derive the result and verify dimensions, scale, and answer format."),
    ],
    "exploiter": [
        ("extract_structure", "Identify the governing relation and all supplied quantities."),
        ("execute_derivation", "Carry out the shortest complete symbolic and numerical derivation."),
        ("verify_result", "Check arithmetic, units, significant figures, and the boxed answer."),
    ],
    "stabilizer": [
        ("preserve_valid_work", "Retain sound intermediate equations from the current solution."),
        ("repair_local_risk", "Repair only the concrete mathematical or formatting weakness."),
        ("consistency_check", "Recheck substitutions, units, precision, and final answer consistency."),
    ],
    "challenger": [
        ("attack_assumptions", "Search for a misread condition, hidden assumption, or missing case."),
        ("stress_test_steps", "Try to falsify each decisive algebraic, calculus, or physical step."),
        ("reconstruct_if_needed", "Repair the derivation only where the adversarial test finds a real flaw."),
    ],
}

CURRICULUM_STAGES: List[Dict[str, Any]] = [
    {
        "level": 1,
        "name": "problem_grounding",
        "difficulty": "easy",
        "objective": "Parse the question, identify knowns and unknowns, and establish one valid governing relation.",
        "constraint": "Avoid speculative alternate methods and long derivations; prioritize correct problem interpretation.",
        "allowed_focus": ["problem_statement", "known_unknown_mapping", "single_governing_relation", "answer_format"],
    },
    {
        "level": 2,
        "name": "basic_consistency",
        "difficulty": "easy",
        "objective": "Preserve grounding while checking units, signs, definitions, and direct substitutions.",
        "constraint": "Add only local consistency checks required by the problem; do not broaden the solution unnecessarily.",
        "allowed_focus": ["units", "signs", "definitions", "substitution", "basic_arithmetic", "answer_format"],
    },
    {
        "level": 3,
        "name": "multi_step_derivation",
        "difficulty": "medium",
        "objective": "Build a complete multi-step derivation and control approximation, precision, and intermediate dependencies.",
        "constraint": "Every added step must contribute to the requested final result and preserve lower-level checks.",
        "allowed_focus": ["algebra", "calculus", "multi_step_reasoning", "approximation", "significant_figures", "units"],
    },
    {
        "level": 4,
        "name": "adversarial_verification",
        "difficulty": "hard",
        "objective": "Stress-test assumptions, alternative interpretations, missing cases, and fragile intermediate steps.",
        "constraint": "Adversarial analysis must repair a concrete risk rather than replace a working derivation with novelty.",
        "allowed_focus": ["counterexample", "edge_cases", "alternative_derivation", "assumption_checks", "error_localization"],
    },
    {
        "level": 5,
        "name": "optimized_finalization",
        "difficulty": "hard",
        "objective": "Refine a reliable solution for correctness, compactness, numerical precision, and unambiguous final presentation.",
        "constraint": "Never sacrifice a stable derivation for cleverness; preserve the simpler solution when evidence is inconclusive.",
        "allowed_focus": ["full_verification", "precision", "compactness", "symbolic_equivalence", "final_answer"],
    },
]

FAILURE_MINIMUM_LEVEL = {
    "problem_misread": 1,
    "answer_format_error": 1,
    "unit_error": 2,
    "algebra_error": 3,
    "calculus_error": 3,
    "approximation_error": 3,
    "missing_case": 4,
    "logical_gap": 4,
}


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    api_retries: int
    retry_sleep: float
    connection_failure_limit: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "APIConfig":
        api_key = args.api_key or os.getenv("HASE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing API key. Use --api-key or set HASE_API_KEY.")
        return cls(
            base_url=str(args.base_url).rstrip("/"),
            api_key=api_key,
            model=str(args.model),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout=float(args.api_timeout),
            api_retries=int(args.api_retries),
            retry_sleep=float(args.retry_sleep),
            connection_failure_limit=int(args.connection_failure_limit),
        )


class APIConnectionLimitError(RuntimeError):
    pass


def extract_api_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("output_text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if isinstance(value, dict):
        return extract_api_text(value.get("text") or value.get("content") or value.get("output_text"))
    return ""


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig):
        self.config = config

    def complete(self, messages: List[Dict[str, str]], request_seed: int) -> Tuple[str, float]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": request_seed,
        }
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
        max_attempts = max(self.config.api_retries + 1, self.config.connection_failure_limit + 1)
        consecutive_connection_failures = 0
        last_error: Optional[BaseException] = None
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
                else:
                    consecutive_connection_failures = 0
                if consecutive_connection_failures >= self.config.connection_failure_limit:
                    raise APIConnectionLimitError(str(last_error)) from exc
                if exc.code not in retryable_http_codes or attempt >= self.config.api_retries:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError, http.client.RemoteDisconnected) as exc:
                last_error = RuntimeError(f"API connection error: {exc}")
                consecutive_connection_failures += 1
                if consecutive_connection_failures >= self.config.connection_failure_limit:
                    raise APIConnectionLimitError(str(last_error)) from exc
            sleep_seconds = self.config.retry_sleep * (attempt + 1)
            print(
                f"[api retry] attempt={attempt + 1}/{max_attempts} sleep={sleep_seconds:.1f}s error={last_error}",
                flush=True,
            )
            time.sleep(sleep_seconds)
        raise RuntimeError(f"API request failed after retries: {last_error}")


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "candidate", "pass", "passed"}:
            return True
        if normalized in {"false", "no", "0", "parent", "fail", "failed"}:
            return False
    return default


def domain_profile_for(type_name: str) -> Dict[str, Any]:
    return DOMAIN_PROFILES[category_for_type(type_name)]


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


JSON_ESCAPE_FOLLOWERS = frozenset('"\\/bfnrtu')
JSON_BACKSLASH_RUN_RE = re.compile(r"(?<!\\)\\+")
JSON_LATEX_COMMAND_RE = re.compile(
    r"(?:"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|"
    r"omicron|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    r"frac|dfrac|tfrac|sqrt|left|right|text|mathrm|textrm|mathbf|mathit|operatorname|"
    r"boxed|fbox|begin|end|overline|underline|bar|vec|hat|dot|ddot|ell|hbar|"
    r"cdot|times|odot|otimes|pm|mp|le|leq|ge|geq|neq|approx|sim|equiv|propto|infty|"
    r"sum|prod|int|iint|iiint|lim|partial|nabla|"
    r"sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp|min|max|det|Re|Im"
    r")(?:\b|(?=\s*\{))"
)


def repair_json_latex_escapes(text: str) -> str:

    def repair_run(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) % 2 == 0:
            return run
        suffix = text[match.end() :]
        next_char = suffix[:1]
        starts_latex_command = bool(JSON_LATEX_COMMAND_RE.match(suffix))
        illegal_json_escape = bool(next_char and next_char not in JSON_ESCAPE_FOLLOWERS)
        if starts_latex_command or illegal_json_escape:
            return run + "\\"
        return run

    return JSON_BACKSLASH_RUN_RE.sub(repair_run, text)


def parse_json(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stripped = strip_json_fence(text)
    repaired = repair_json_latex_escapes(stripped)
    try:
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else dict(default or {})
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(repaired):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(repaired[index:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return dict(default or {})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def minimum_curriculum_level(predicted_modes: Sequence[str]) -> int:
    return max([FAILURE_MINIMUM_LEVEL.get(str(mode), 1) for mode in predicted_modes] or [1])


def select_curriculum_stage(
    round_index: int,
    sample_index: int,
    samples_per_round: int,
    last_mirror_feedback: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stage_count = len(CURRICULUM_STAGES)
    sample_count = max(1, samples_per_round)
    sample_level = 1 + int(sample_index * stage_count / sample_count)
    round_level = min(stage_count, max(1, round_index))
    proposed_level = max(1, min(stage_count, max(sample_level, round_level)))

    predicted_correctness = 0.0
    severity_score = 0.0
    predicted_modes: List[str] = []
    if last_mirror_feedback:
        predicted_correctness = clamp(last_mirror_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)
        severity_score = clamp(last_mirror_feedback.get("severity_score"), 0.0, 1.0, 1.0)
        predicted_modes = [str(mode) for mode in last_mirror_feedback.get("predicted_failure_modes", [])]

    repair_min_level = minimum_curriculum_level(predicted_modes)
    selected_level = proposed_level
    gate_reason = "round_and_sample_progression"
    if last_mirror_feedback and (predicted_correctness < 0.35 or severity_score >= 0.75):
        selected_level = min(proposed_level, max(2, repair_min_level))
        gate_reason = (
            "mirror_high_risk_repair_requires_advanced_reasoning"
            if repair_min_level > 2
            else "mirror_high_risk_simplify_before_upgrade"
        )
    elif last_mirror_feedback and predicted_correctness >= 0.75 and severity_score <= 0.25:
        selected_level = min(stage_count, proposed_level + 1)
        gate_reason = "mirror_predicted_reliable_upgrade"
    elif last_mirror_feedback:
        selected_level = min(proposed_level, max(3, repair_min_level))
        gate_reason = (
            "mirror_medium_risk_repair_requires_advanced_reasoning"
            if repair_min_level > 3
            else "mirror_medium_risk_hold_at_multistep_boundary"
        )

    stage = dict(CURRICULUM_STAGES[selected_level - 1])
    stage.update(
        {
            "stage": stage["difficulty"],
            "round": round_index,
            "sample_index": sample_index,
            "samples_per_round": samples_per_round,
            "proposed_level": proposed_level,
            "selected_level": selected_level,
            "gate_reason": gate_reason,
            "predicted_correctness": predicted_correctness,
            "severity_score": severity_score,
            "predicted_failure_modes": predicted_modes,
            "repair_min_level": repair_min_level,
            "progression_rule": (
                "Begin with problem grounding, increase to consistency and multi-step derivation, "
                "then add adversarial verification and optimized finalization as mirror risk permits."
            ),
        }
    )
    return stage


def choose_macro_role(
    round_index: int,
    type_name: str,
    mirror_history: List[Dict[str, Any]],
    role_stats: Dict[str, Dict[str, float]],
    role_history: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    category = category_for_type(type_name)
    domain_profile = domain_profile_for(type_name)
    last_feedback = mirror_history[-1] if mirror_history else None
    recommended_role = ""
    predicted_modes: List[str] = []
    confidence = 0.0
    predicted_correctness = 0.0
    if last_feedback:
        recommended_role = str(last_feedback.get("recommended_macro_role", "")).lower().strip()
        predicted_modes = [str(mode) for mode in last_feedback.get("predicted_failure_modes", [])]
        confidence = clamp(last_feedback.get("confidence"), 0.0, 1.0, 0.0)
        predicted_correctness = clamp(last_feedback.get("predicted_correctness"), 0.0, 1.0, 0.0)

    total_trials = sum(role_stats[role]["trials"] for role in MACRO_ROLES)
    recent_roles = role_history[-3:]
    mode_set = set(predicted_modes)
    role_scores: Dict[str, Dict[str, float]] = {}
    cold_start = {"exploiter": 0.035, "challenger": 0.020, "explorer": 0.015, "stabilizer": 0.0}
    for role in MACRO_ROLES:
        stats = role_stats[role]
        trials = stats["trials"]
        historical_success = 0.5 if trials == 0 else stats["success"] / max(1.0, trials)
        exploration_bonus = 0.34 if trials == 0 else 0.18 * ((total_trials + 1.0) ** 0.5) / ((trials + 1.0) ** 0.5)
        mirror_bonus = 0.22 + 0.18 * confidence if recommended_role == role else 0.0
        failure_fit_bonus = 0.0
        if role == "stabilizer" and mode_set & {"algebra_error", "unit_error", "approximation_error", "answer_format_error"}:
            failure_fit_bonus += 0.16
        if role == "challenger" and mode_set & {"problem_misread", "missing_case", "logical_gap"}:
            failure_fit_bonus += 0.15
        if role == "explorer" and mode_set & {"problem_misread", "logical_gap", "calculus_error"}:
            failure_fit_bonus += 0.12
        if role == "exploiter" and last_feedback and predicted_correctness >= 0.75 and not mode_set:
            failure_fit_bonus += 0.10
        novelty_bonus = 0.10 if role not in recent_roles else 0.0
        domain_prior = float(domain_profile.get("role_prior", {}).get(role, 0.0))
        repetition_penalty = 0.13 * recent_roles.count(role)
        if role_history and role_history[-1] == role:
            repetition_penalty += 0.10
        score = (
            0.42 * historical_success
            + exploration_bonus
            + mirror_bonus
            + failure_fit_bonus
            + novelty_bonus
            + domain_prior
            + cold_start[role]
            - repetition_penalty
        )
        role_scores[role] = {
            "score": score,
            "historical_success": historical_success,
            "exploration_bonus": exploration_bonus,
            "mirror_bonus": mirror_bonus,
            "failure_fit_bonus": failure_fit_bonus,
            "novelty_bonus": novelty_bonus,
            "domain_prior": domain_prior,
            "repetition_penalty": repetition_penalty,
        }

    selected = max(MACRO_ROLES, key=lambda role: (role_scores[role]["score"], -role_stats[role]["trials"], role))
    late_override = False
    original_role = selected
    if (
        args.enable_late_stabilizer
        and round_index >= args.late_stability_round
        and last_feedback
        and predicted_correctness >= args.late_stability_correctness
        and clamp(last_feedback.get("severity_score"), 0.0, 1.0, 1.0) <= args.late_stability_severity
        and mode_set & {"algebra_error", "unit_error", "approximation_error", "answer_format_error", "logical_gap"}
        and selected in {"challenger", "explorer"}
    ):
        selected = "stabilizer"
        late_override = True

    reason = (
        f"Dynamic exploration scoring selected {original_role} from mirror feedback, historical "
        "role performance, exploration, failure-mode fit, novelty, and repetition penalties."
    )
    if late_override:
        reason += " Late-stage stability control replaced it with stabilizer to preserve a coherent derivation."
    return {
        "role": selected,
        "reason": reason,
        "role_instruction": ROLE_INSTRUCTIONS[selected],
        "signals": {
            "source": "dynamic_exploration_scoring",
            "category": category,
            "domain_reasoning_focus": domain_profile["reasoning_focus"],
            "domain_required_verification": domain_profile["required_verification"],
            "recommended_macro_role": recommended_role,
            "predicted_failure_modes": predicted_modes,
            "mirror_confidence": confidence,
            "predicted_correctness": predicted_correctness,
            "recent_roles": recent_roles,
            "role_scores": role_scores,
            "late_stability_override": late_override,
            "original_role": original_role,
        },
    }


def normalize_step_id(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return normalized or fallback


def fallback_reasoning_strategy(macro: Dict[str, Any], curriculum: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = ROLE_FALLBACK_STEPS.get(macro["role"], ROLE_FALLBACK_STEPS["exploiter"])
    return [
        {
            "step_index": index,
            "step_id": step_id,
            "step_name": step_id.replace("_", " "),
            "instruction": instruction,
            "base_instruction": instruction,
            "expected_private_output": "a mathematical intermediate result or verification decision",
            "macro_direction": macro["role"],
            "curriculum_level": curriculum["level"],
            "strategy_source": "fallback",
        }
        for index, (step_id, instruction) in enumerate(steps, start=1)
    ]


def parse_reasoning_strategy(raw_text: str, macro: Dict[str, Any], curriculum: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = parse_json(raw_text, {})
    raw_steps = parsed.get("reasoning_steps") or parsed.get("steps") or []
    if not isinstance(raw_steps, list):
        return fallback_reasoning_strategy(macro, curriculum)
    strategy: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            continue
        step_id = normalize_step_id(str(item.get("step_id") or item.get("step_name") or ""), f"step_{index}")
        strategy.append(
            {
                "step_index": index,
                "step_id": step_id,
                "step_name": str(item.get("step_name") or step_id.replace("_", " ")),
                "instruction": instruction,
                "base_instruction": instruction,
                "expected_private_output": str(item.get("expected_private_output") or "a mathematical decision"),
                "macro_direction": macro["role"],
                "curriculum_level": curriculum["level"],
                "strategy_rationale": str(parsed.get("strategy_rationale", "")),
                "strategy_source": "model",
            }
        )
    return strategy or fallback_reasoning_strategy(macro, curriculum)


def apply_step_feedback(strategy: List[Dict[str, Any]], mirror: Dict[str, Any]) -> List[Dict[str, Any]]:
    feedback_by_step = {
        str(item.get("step_id")): item
        for item in mirror.get("step_feedback", [])
        if isinstance(item, dict) and item.get("step_id")
    }
    optimized: List[Dict[str, Any]] = []
    for step in strategy:
        updated = dict(step)
        feedback = feedback_by_step.get(str(step.get("step_id")))
        if feedback:
            updated["mirror_optimized"] = True
            updated["mirror_risk"] = str(feedback.get("risk", ""))
            updated["instruction"] = str(feedback.get("updated_instruction") or step["instruction"])
        else:
            updated["mirror_optimized"] = False
            updated["mirror_risk"] = ""
        optimized.append(updated)
    return optimized


def normalize_candidate(raw_text: str, parsed: Dict[str, Any], previous_solution: str = "") -> Dict[str, Any]:
    reasoning = str(parsed.get("reasoning") or parsed.get("derivation") or "").strip()
    solution = str(parsed.get("solution") or parsed.get("final_solution") or "").strip()
    final_answer = str(parsed.get("final_answer") or parsed.get("answer") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if not solution:
        solution = reasoning or raw_text.strip()
    if not final_answer:
        final_answer = extract_final_answer(solution)
    elif r"\boxed" in final_answer or r"\fbox" in final_answer:
        final_answer = extract_final_answer(final_answer)
    if final_answer and r"\boxed" not in solution:
        solution = f"{solution.rstrip()}\n\nFinal answer: \\boxed{{{final_answer}}}".strip()
    if not solution and previous_solution:
        solution = previous_solution
        final_answer = extract_final_answer(previous_solution)
        rationale = "Model output was empty; retained the previous non-empty solution."
    return {
        "solution": solution,
        "final_answer": final_answer,
        "rationale": rationale,
        "repair_scope": str(parsed.get("repair_scope", "")).strip().lower(),
        "preserved_steps": parsed.get("preserved_steps", []) if isinstance(parsed.get("preserved_steps", []), list) else [],
        "answer_change_justification": str(parsed.get("answer_change_justification", "")).strip(),
        "independent_check": str(parsed.get("independent_check", "")).strip(),
        "non_empty": bool(solution.strip()),
        "has_final_answer": bool(final_answer.strip()),
    }


def apply_late_solution_damper(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    curriculum: Dict[str, Any],
    round_index: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    decision = {"applied": False, "reason": "late_solution_damper_not_triggered"}
    if (
        not args.enable_solution_drift_damper
        or round_index < args.late_stability_round
        or int(curriculum.get("level", 0)) < 4
        or not previous.get("solution")
    ):
        return current, decision
    previous_text = str(previous.get("solution", ""))
    current_text = str(current.get("solution", ""))
    if not current.get("has_final_answer"):
        return dict(previous), {"applied": True, "reason": "late_candidate_missing_final_answer"}
    minimum_length = max(args.late_min_solution_chars, int(len(previous_text) * args.late_min_solution_ratio))
    if len(current_text) < minimum_length:
        return dict(previous), {
            "applied": True,
            "reason": "late_candidate_is_structurally_truncated",
            "candidate_length": len(current_text),
            "minimum_length": minimum_length,
        }
    return current, decision


def build_strategy_prompt(
    problem: str,
    type_name: str,
    macro: Dict[str, Any],
    curriculum: Dict[str, Any],
    current_solution: Dict[str, Any],
    last_mirror: Optional[Dict[str, Any]],
) -> str:
    domain_profile = domain_profile_for(type_name)
    parent_available = bool(current_solution.get("solution") or current_solution.get("final_answer"))
    evolution_mode = "conservative parent-preserving update" if parent_available else "initial independent solve"
    return f"""
You are the same Minerva Math agent planning its private reasoning process. Return strict JSON only.
Do not solve from a reference answer; no reference solution is available during evolution.

Evolution mode: {evolution_mode}

{MINERVA_EVIDENCE_HIERARCHY}

{PARENT_ANSWER_CHANGE_GATE if parent_available else "No parent is available. Build the simplest complete solution directly from the problem statement."}

Problem source: {type_name}
Problem:
{problem}

Minerva domain profile:
{json.dumps(domain_profile, ensure_ascii=False, indent=2)}

Macro role: {macro['role']}
Macro reason: {macro['reason']}
Macro instruction: {macro['role_instruction']}

Curriculum stage:
{json.dumps(curriculum, ensure_ascii=False, indent=2)}

Current inherited solution, which may be empty in round 1:
{current_solution.get('solution', '')}

Previous mirror feedback:
{json.dumps(last_mirror or {}, ensure_ascii=False, indent=2)}

Create only the reasoning steps useful for this problem, role, and curriculum stage.
Choose between 2 and 7 steps. Use unique snake_case step_id values.
At least one step must perform the domain-required verification rather than merely restating the derivation.
When a parent exists, the first step must audit one decisive parent equation and the final step must decide preserve_parent or change_parent using the three-part answer-change gate.
Changing macro role changes the verification focus, not the burden of proof required to change the inherited answer.
Return exactly:
{{
  "strategy_rationale": "why this sequence fits the macro role and current mirror risk",
  "reasoning_steps": [
    {{
      "step_id": "unique_snake_case_id",
      "step_name": "short readable name",
      "instruction": "private mathematical reasoning instruction",
      "expected_private_output": "intermediate result or verification decision"
    }}
  ]
}}

{STRICT_JSON_PROMPT_RULES}
""".strip()


def build_generation_prompt(
    problem: str,
    type_name: str,
    round_index: int,
    max_rounds: int,
    macro: Dict[str, Any],
    curriculum: Dict[str, Any],
    current_solution: Dict[str, Any],
    mirror_history: List[Dict[str, Any]],
    strategy: List[Dict[str, Any]],
) -> str:
    domain_profile = domain_profile_for(type_name)
    parent_available = bool(current_solution.get("solution") or current_solution.get("final_answer"))
    evolution_mode = "conservative parent-preserving update" if parent_available else "initial independent solve"
    return f"""
Solve this Minerva Math problem using HASE-v1 hierarchical self-evolution.
The official metric is final-answer Accuracy, but benchmark/reference answers are unavailable and must not be guessed or used during evolution.

Round: {round_index}/{max_rounds}
Evolution mode: {evolution_mode}
Problem source: {type_name}
Problem:
{problem}

Minerva domain profile:
{json.dumps(domain_profile, ensure_ascii=False, indent=2)}

Macro role: {macro['role']}
Macro reason: {macro['reason']}
Macro instruction: {macro['role_instruction']}

Curriculum stage:
{json.dumps(curriculum, ensure_ascii=False, indent=2)}

Current inherited solution:
{current_solution.get('solution', '')}

Inherited final answer:
{current_solution.get('final_answer', '')}

{MINERVA_EVIDENCE_HIERARCHY}

{PARENT_ANSWER_CHANGE_GATE if parent_available else "No parent is available. Solve directly and verify the final answer once."}

Recent non-oracle mirror feedback:
{json.dumps(mirror_history[-3:], ensure_ascii=False, indent=2)}

Dynamic reasoning strategy:
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Follow the curriculum stage and macro direction. Produce a complete derivation, check units and requested precision, and put exactly one final answer in \\boxed{{}}.
Apply the domain-required verification before committing to the final answer. Preserve exact symbolic forms until the problem requests a numerical approximation.
For rounds after the initial solve, begin from verified parent steps instead of solving from scratch. If the answer-change gate does not pass, copy the inherited final answer exactly.
Never replace a prompt-provided approximation, material constant, cross-section convention, or completed subproblem result with a preferred external convention.
Do not mention benchmark evaluation or a reference answer.

Return only one JSON object:
{{
  "rationale": "specific reason for this round's reasoning update",
  "reasoning": "complete mathematical derivation",
  "final_answer": "content that belongs inside the final box",
  "solution": "complete derivation ending in exactly one \\\\boxed{{...}}",
  "repair_scope": "none|local|global",
  "preserved_steps": ["verified inherited step retained unchanged"],
  "answer_change_justification": "unchanged, or exact localized evidence satisfying all three answer-change conditions",
  "independent_check": "compact domain-specific verification"
}}
 
{STRICT_JSON_PROMPT_RULES}
""".strip()


def build_mirror_prompt(
    problem: str,
    type_name: str,
    candidate: Dict[str, Any],
    macro: Dict[str, Any],
    curriculum: Dict[str, Any],
    strategy: List[Dict[str, Any]],
    parent_solution: Optional[Dict[str, Any]] = None,
    comparison_scope: str = "no_parent_available",
) -> str:
    domain_profile = domain_profile_for(type_name)
    parent_solution = parent_solution or {}
    return f"""
You are the mirror adversary for a Minerva Math self-evolving agent.
Critique the candidate without access to the reference solution or official correctness result.
Use mathematical verification, dimensional analysis, counterfactual reasoning, and independent recomputation.

{MINERVA_EVIDENCE_HIERARCHY}

{PARENT_ANSWER_CHANGE_GATE}

Problem source: {type_name}
Problem:
{problem}

Minerva domain profile:
{json.dumps(domain_profile, ensure_ascii=False, indent=2)}

Macro role: {macro['role']}
Curriculum stage:
{json.dumps(curriculum, ensure_ascii=False, indent=2)}

Candidate solution:
{candidate.get('solution', '')}

Candidate extracted final answer:
{candidate.get('final_answer', '')}

Comparison scope: {comparison_scope}
Parent/reference candidate from the previous evolution step, which may be empty:
{parent_solution.get('solution', '')}

Parent extracted final answer:
{parent_solution.get('final_answer', '')}

Reasoning strategy used by the generator:
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Critique the normalized candidate above. Do not assume that confidence or fluent writing implies correctness.
Independently solve or verify the decisive equation and report the independently obtained answer.
When a parent is present, it is the default preferred solution. Prefer the candidate only if it fixes a localized parent error and the independent answer supports the candidate's final answer.
If candidate and parent answers differ, quote the exact prompt evidence and decisive equation that prove which one is valid. A general textbook convention or remembered empirical value is insufficient.
If the candidate is truncated, malformed, or lacks a complete final answer, prefer a complete parent regardless of the candidate's partial reasoning.
If no concrete parent error is found, set preferred_solution to parent, answer_change_required to false, repair_scope to none or local, and predicted_failure_modes to an empty list.
Do not reward novelty. A failed adversarial attack is evidence for preserving the parent.
Return only JSON:
{{
  "adversarial_summary": "short mathematical critique",
  "predicted_correctness": 0.0,
  "severity_score": 0.0,
  "confidence": 0.0,
  "reliability": 0.0,
  "predicted_failure_modes": ["algebra_error", "unit_error"],
  "strengths": ["specific verified strength"],
  "weaknesses": ["specific mathematical weakness"],
  "independent_check": "brief independent recomputation or invariant check",
  "independent_answer": "answer obtained by the independent check",
  "domain_verification_passed": true,
  "domain_verification_details": "units/symbolic conditions/assumptions/precision checks required by this category",
  "preferred_solution": "candidate|parent",
  "pairwise_confidence": 0.0,
  "candidate_improvements": ["specific improvement over the parent"],
  "candidate_regressions": ["specific regression relative to the parent"],
  "error_localization": "exact equation or reasoning step that is wrong, or none",
  "answer_change_required": false,
  "answer_change_confidence": 0.0,
  "repair_scope": "none|local|global",
  "verified_steps": ["step that should be preserved"],
  "suggested_update": "specific executable repair",
  "repair_strategy": "one concise rule for the next reasoning attempt",
  "curriculum_feedback": "simplify, hold, or increase complexity",
  "step_feedback": [
    {{
      "step_id": "one exact step_id from the strategy",
      "risk": "what is mathematically weak",
      "updated_instruction": "how this exact step should be optimized"
    }}
  ],
  "recommended_macro_role": "explorer|exploiter|stabilizer|challenger",
  "overall_score": 0.0
}}
All numeric confidence-like fields must be between 0 and 1.
Each predicted_failure_modes item must be selected from: problem_misread, algebra_error, calculus_error, unit_error, approximation_error, missing_case, logical_gap, answer_format_error. Return an empty list when no concrete failure is found.
{STRICT_JSON_PROMPT_RULES}
""".strip()


def build_micro_prompt(
    problem: str,
    type_name: str,
    candidate: Dict[str, Any],
    mirror: Dict[str, Any],
    macro: Dict[str, Any],
    curriculum: Dict[str, Any],
    optimized_strategy: List[Dict[str, Any]],
) -> str:
    domain_profile = domain_profile_for(type_name)
    return f"""
You are the micro-optimizer inside {METHOD_NAME}.
Repair the candidate at the level of individual reasoning steps using mirror feedback.
Do not use or infer a hidden reference answer.

{MINERVA_EVIDENCE_HIERARCHY}

Problem source: {type_name}
Problem:
{problem}

Minerva domain profile:
{json.dumps(domain_profile, ensure_ascii=False, indent=2)}

Macro role: {macro['role']}
Curriculum stage:
{json.dumps(curriculum, ensure_ascii=False, indent=2)}

Pre-micro candidate:
{candidate.get('solution', '')}

Mirror feedback:
{json.dumps(mirror, ensure_ascii=False, indent=2)}

Mirror-optimized reasoning strategy:
{json.dumps(optimized_strategy, ensure_ascii=False, indent=2)}

Make the smallest complete repair that addresses concrete mathematical risks. Preserve verified steps.
Check units, requested significant figures, and final answer format. End in exactly one \\boxed{{}}.
Do not change the final answer merely because the mirror expresses uncertainty. Change it only when the mirror localizes a concrete error and its independent check supports a replacement.
Treat mirror feedback as a hypothesis, not authority. Reject feedback that conflicts with explicit problem data, reopens completed subproblems, or relies only on an alternative external convention.
If error_localization is none, answer_change_required is false, or the independent answer does not support a replacement, return the candidate answer unchanged.
If the candidate output is malformed but its extracted answer is coherent, reconstruct a compact complete solution around that same answer instead of inventing a new one.
For physics and chemistry, preserve exact values until the requested rounding step. For mathematics, preserve constants, domains, branches, and initial conditions. For economics/ecology, preserve definitions and model assumptions.

Return only one JSON object:
{{
  "rationale": "specific micro-level repair",
  "reasoning": "complete repaired derivation",
  "final_answer": "content inside the final box",
  "solution": "complete repaired derivation ending in exactly one \\\\boxed{{...}}",
  "repair_scope": "none|local|global",
  "preserved_steps": ["verified step retained unchanged"],
  "answer_change_justification": "localized evidence for changing the answer, or unchanged",
  "independent_check": "domain-specific verification performed after repair"
}}
{STRICT_JSON_PROMPT_RULES}
""".strip()


def normalize_mirror_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    raw_modes = payload.get("predicted_failure_modes", [])
    if isinstance(raw_modes, str):
        raw_modes = re.split(r"[|,;]", raw_modes)
    modes: List[str] = []
    if isinstance(raw_modes, list):
        for item in raw_modes:
            for mode in re.split(r"[|,;]", str(item)):
                candidate = mode.strip()
                if candidate in MATH_FAILURE_MODES and candidate not in modes:
                    modes.append(candidate)
    normalized["predicted_failure_modes"] = modes
    normalized["predicted_correctness"] = clamp(payload.get("predicted_correctness"), 0.0, 1.0, 0.0)
    normalized["severity_score"] = clamp(payload.get("severity_score"), 0.0, 1.0, 1.0)
    normalized["confidence"] = clamp(payload.get("confidence"), 0.0, 1.0, 0.0)
    normalized["reliability"] = clamp(payload.get("reliability"), 0.0, 1.0, 0.0)
    normalized["overall_score"] = clamp(payload.get("overall_score"), 0.0, 1.0, 0.0)
    recommended_role = str(payload.get("recommended_macro_role", "")).strip().lower()
    normalized["recommended_macro_role"] = recommended_role if recommended_role in MACRO_ROLES else ""
    step_feedback = payload.get("step_feedback", [])
    normalized["step_feedback"] = step_feedback if isinstance(step_feedback, list) else []
    preferred_solution = str(payload.get("preferred_solution", "")).strip().lower()
    normalized["preferred_solution"] = preferred_solution if preferred_solution in {"candidate", "parent"} else ""
    normalized["pairwise_confidence"] = clamp(payload.get("pairwise_confidence"), 0.0, 1.0, 0.0)
    normalized["domain_verification_passed"] = as_bool(payload.get("domain_verification_passed"), False)
    normalized["independent_answer"] = str(payload.get("independent_answer", "")).strip()
    normalized["answer_change_required"] = as_bool(payload.get("answer_change_required"), False)
    normalized["answer_change_confidence"] = clamp(payload.get("answer_change_confidence"), 0.0, 1.0, 0.0)
    repair_scope = str(payload.get("repair_scope", "")).strip().lower()
    normalized["repair_scope"] = repair_scope if repair_scope in {"none", "local", "global"} else ""
    normalized["error_localization"] = str(payload.get("error_localization", "")).strip()
    normalized["verified_steps"] = payload.get("verified_steps", []) if isinstance(payload.get("verified_steps", []), list) else []
    return normalized


def mirror_selection_score(mirror: Dict[str, Any]) -> Tuple[float, float, float]:
    correctness = clamp(mirror.get("predicted_correctness"), 0.0, 1.0, 0.0)
    severity = clamp(mirror.get("severity_score"), 0.0, 1.0, 1.0)
    confidence = clamp(mirror.get("confidence"), 0.0, 1.0, 0.0)
    failure_count = len(mirror.get("predicted_failure_modes", []))
    return correctness, -severity, confidence - 0.05 * failure_count


def post_micro_guard_decision(
    pre_mirror: Dict[str, Any],
    post_mirror: Dict[str, Any],
    pre_candidate: Dict[str, Any],
    micro_candidate: Dict[str, Any],
    category: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pre_score = mirror_selection_score(pre_mirror)
    post_score = mirror_selection_score(post_mirror)
    pre_correctness = pre_score[0]
    post_correctness = post_score[0]
    pre_severity = -pre_score[1]
    post_severity = -post_score[1]
    correctness_regressed = post_correctness < pre_correctness - args.post_micro_guard_correctness_margin
    severity_regressed = post_severity > pre_severity + args.post_micro_guard_severity_margin
    confidence_regressed = post_score[2] < pre_score[2] - args.post_micro_guard_margin
    component_regressed = any(
        post < pre - args.post_micro_guard_margin for pre, post in zip(pre_score, post_score)
    )
    pre_modes = {str(mode) for mode in pre_mirror.get("predicted_failure_modes", [])}
    post_modes = {str(mode) for mode in post_mirror.get("predicted_failure_modes", [])}
    new_blocking_modes = sorted((post_modes - pre_modes) & MATH_FAILURE_MODES)
    answer_comparison = answers_equivalent(
        str(pre_candidate.get("final_answer", "")),
        str(micro_candidate.get("final_answer", "")),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    answer_changed = not bool(answer_comparison.get("correct"))
    independent_answer_comparison = answers_equivalent(
        str(post_mirror.get("independent_answer", "")),
        str(micro_candidate.get("final_answer", "")),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    error_localization = str(pre_mirror.get("error_localization", "")).strip().lower()
    localized_error_present = bool(error_localization and error_localization not in {"none", "no error", "n/a"})
    answer_change_supported = bool(
        pre_mirror.get("answer_change_required")
        and clamp(pre_mirror.get("answer_change_confidence"), 0.0, 1.0, 0.0)
        >= args.micro_answer_change_min_confidence
        and localized_error_present
        and independent_answer_comparison.get("correct")
        and post_mirror.get("domain_verification_passed")
        and str(micro_candidate.get("answer_change_justification", "")).strip()
    )
    unsupported_answer_change = answer_changed and not answer_change_supported
    category_verification_failed = bool(
        answer_changed
        and category in {"chemistry", "mathematics", "physics", "other"}
        and not post_mirror.get("domain_verification_passed")
    )
    accepted = not (
        correctness_regressed
        or severity_regressed
        or confidence_regressed
        or component_regressed
        or new_blocking_modes
        or unsupported_answer_change
        or category_verification_failed
    )
    if accepted:
        reason = "post_micro_solution_passed_componentwise_mirror_guard"
    elif correctness_regressed:
        reason = "post_micro_predicted_correctness_regressed"
    elif severity_regressed:
        reason = "post_micro_severity_increased"
    elif confidence_regressed:
        reason = "post_micro_confidence_component_regressed"
    elif new_blocking_modes:
        reason = "post_micro_introduced_new_blocking_failure_modes"
    elif unsupported_answer_change:
        reason = "post_micro_answer_change_lacks_localized_independent_evidence"
    elif category_verification_failed:
        reason = "post_micro_answer_change_failed_domain_verification"
    else:
        reason = "post_micro_componentwise_mirror_score_regressed"
    return {
        "accepted": accepted,
        "reason": reason,
        "pre_micro_score": list(pre_score),
        "post_micro_score": list(post_score),
        "new_blocking_failure_modes": new_blocking_modes,
        "category": category,
        "answer_changed": answer_changed,
        "answer_comparison_method": answer_comparison.get("match_method"),
        "answer_change_supported": answer_change_supported,
        "localized_error_present": localized_error_present,
        "independent_answer_matches_micro": bool(independent_answer_comparison.get("correct")),
        "independent_answer_match_method": independent_answer_comparison.get("match_method"),
        "domain_verification_passed": bool(post_mirror.get("domain_verification_passed")),
        "thresholds": {
            "score_margin": args.post_micro_guard_margin,
            "correctness_margin": args.post_micro_guard_correctness_margin,
            "severity_margin": args.post_micro_guard_severity_margin,
            "answer_change_min_confidence": args.micro_answer_change_min_confidence,
        },
    }


def parent_inheritance_guard_decision(
    parent: Dict[str, Any],
    candidate: Dict[str, Any],
    pre_mirror: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not parent.get("solution"):
        return {
            "candidate_accepted": True,
            "reason": "no_inherited_parent_available",
            "answer_changed_from_parent": False,
        }
    answer_comparison = answers_equivalent(
        str(candidate.get("final_answer", "")),
        str(parent.get("final_answer", "")),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    answer_changed = not bool(answer_comparison.get("correct"))
    preferred_solution = str(pre_mirror.get("preferred_solution", ""))
    pairwise_confidence = clamp(pre_mirror.get("pairwise_confidence"), 0.0, 1.0, 0.0)
    explicit_parent_preference = bool(
        preferred_solution == "parent" and pairwise_confidence >= args.parent_guard_confidence
    )
    unsupported_parent_answer_change = bool(
        answer_changed
        and not pre_mirror.get("answer_change_required")
        and not pre_mirror.get("domain_verification_passed")
    )
    accepted = not (explicit_parent_preference or unsupported_parent_answer_change)
    if explicit_parent_preference:
        reason = "pairwise_mirror_prefers_inherited_parent"
    elif unsupported_parent_answer_change:
        reason = "candidate_changes_parent_answer_without_domain_evidence"
    else:
        reason = "candidate_passed_parent_inheritance_guard"
    return {
        "candidate_accepted": accepted,
        "reason": reason,
        "answer_changed_from_parent": answer_changed,
        "answer_comparison_method": answer_comparison.get("match_method"),
        "preferred_solution": preferred_solution,
        "pairwise_confidence": pairwise_confidence,
        "domain_verification_passed": bool(pre_mirror.get("domain_verification_passed")),
        "threshold": args.parent_guard_confidence,
    }


def mirror_reliability_evidence(
    mirror: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    correctness = clamp(mirror.get("predicted_correctness"), 0.0, 1.0, 0.0)
    severity = clamp(mirror.get("severity_score"), 0.0, 1.0, 1.0)
    confidence = clamp(mirror.get("confidence"), 0.0, 1.0, 0.0)
    modes = [str(mode) for mode in mirror.get("predicted_failure_modes", [])]
    blocking = sorted(set(modes) & set(args.mirror_stop_blocking_modes))
    independent_check = str(mirror.get("independent_check", "")).strip()
    reliable = bool(
        correctness >= args.mirror_stop_correctness
        and severity <= args.mirror_stop_severity
        and confidence >= args.mirror_stop_confidence
        and not blocking
        and len(independent_check) >= args.mirror_stop_min_independent_check_chars
    )
    return {
        "reliable": reliable,
        "predicted_correctness": correctness,
        "severity_score": severity,
        "confidence": confidence,
        "predicted_failure_modes": modes,
        "blocking_failure_modes": blocking,
        "independent_check_length": len(independent_check),
    }


def mirror_stop_decision(
    final_mirror: Dict[str, Any],
    pre_mirror: Dict[str, Any],
    post_mirror: Dict[str, Any],
    guard_decision: Dict[str, Any],
    round_index: int,
    curriculum: Dict[str, Any],
    previous_answer: str,
    candidate_answer: str,
    pre_micro_answer: str,
    micro_answer: str,
    inherited_parent: bool,
    reliable_streak: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    final_evidence = mirror_reliability_evidence(final_mirror, args)
    pre_evidence = mirror_reliability_evidence(pre_mirror, args)
    post_evidence = mirror_reliability_evidence(post_mirror, args)
    cross_round_comparison = answers_equivalent(
        candidate_answer,
        previous_answer,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    ) if previous_answer and candidate_answer else {
        "correct": False,
        "match_method": "missing_previous_or_candidate_answer",
    }
    within_round_comparison = answers_equivalent(
        pre_micro_answer,
        micro_answer,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    ) if pre_micro_answer and micro_answer else {
        "correct": False,
        "match_method": "missing_pre_or_micro_answer",
    }
    cross_round_stable = bool(cross_round_comparison.get("correct"))
    within_round_stable = bool(within_round_comparison.get("correct"))
    answer_stable = cross_round_stable if previous_answer else within_round_stable
    independent_answer_comparison = answers_equivalent(
        str(final_mirror.get("independent_answer", "")),
        candidate_answer,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    independent_answer_matches = bool(independent_answer_comparison.get("correct"))
    correctness_gap = abs(
        pre_evidence["predicted_correctness"] - post_evidence["predicted_correctness"]
    )
    severity_gap = abs(pre_evidence["severity_score"] - post_evidence["severity_score"])
    pre_post_agree = bool(
        pre_evidence["reliable"]
        and post_evidence["reliable"]
        and correctness_gap <= args.mirror_stop_max_correctness_gap
        and severity_gap <= args.mirror_stop_max_severity_gap
    )
    guard_ready = bool(guard_decision.get("accepted"))
    answer_ready = answer_stable or not args.mirror_stop_require_answer_stability
    agreement_ready = pre_post_agree or not args.mirror_stop_require_pre_post_agreement
    independent_ready = (
        independent_answer_matches or not args.mirror_stop_require_independent_answer_match
    )
    domain_ready = bool(final_mirror.get("domain_verification_passed")) or not args.mirror_stop_require_domain_verification
    evidence_ready = bool(
        final_evidence["reliable"]
        and guard_ready
        and answer_ready
        and agreement_ready
        and independent_ready
        and domain_ready
        and not inherited_parent
    )
    next_streak = reliable_streak + 1 if evidence_ready else 0
    should_stop = bool(
        args.mirror_stop
        and next_streak >= args.mirror_stop_consecutive_rounds
    )

    if not args.mirror_stop:
        reason = "mirror_stop_disabled"
    elif inherited_parent:
        reason = "candidate_rejected_by_parent_inheritance_guard"
    elif not guard_ready:
        reason = "post_micro_guard_did_not_accept_candidate"
    elif not final_evidence["reliable"]:
        reason = "final_mirror_reliability_evidence_insufficient"
    elif not answer_ready:
        reason = "final_answer_not_stable_across_rounds"
    elif not agreement_ready:
        reason = "pre_post_mirror_evidence_not_consistent"
    elif not independent_ready:
        reason = "independent_answer_does_not_match_candidate"
    elif not domain_ready:
        reason = "domain_specific_verification_not_passed"
    elif next_streak < args.mirror_stop_consecutive_rounds:
        reason = "reliable_streak_below_required_length"
    else:
        reason = "domain_aware_mirror_consensus_predicts_solution_is_reliable"
    return {
        "should_stop": should_stop,
        "reason": reason,
        "reliable_streak_before": reliable_streak,
        "reliable_streak_after": next_streak,
        "required_consecutive_rounds": args.mirror_stop_consecutive_rounds,
        "answer_stable": answer_stable,
        "cross_round_answer_stable": cross_round_stable,
        "within_round_answer_stable": within_round_stable,
        "answer_stability_match_method": (
            cross_round_comparison.get("match_method")
            if previous_answer
            else within_round_comparison.get("match_method")
        ),
        "independent_answer_matches_candidate": independent_answer_matches,
        "independent_answer_match_method": independent_answer_comparison.get("match_method"),
        "domain_verification_passed": bool(final_mirror.get("domain_verification_passed")),
        "inherited_parent": inherited_parent,
        "pre_post_mirror_agree": pre_post_agree,
        "pre_post_correctness_gap": correctness_gap,
        "pre_post_severity_gap": severity_gap,
        "post_micro_guard_accepted": guard_ready,
        "official_metric_gate_enabled": False,
        "official_metric_correct": None,
        "curriculum_level": int(curriculum.get("level", 0)),
        "final_mirror_evidence": final_evidence,
        "pre_mirror_evidence": pre_evidence,
        "post_mirror_evidence": post_evidence,
        "thresholds": {
            "predicted_correctness": args.mirror_stop_correctness,
            "maximum_severity": args.mirror_stop_severity,
            "minimum_confidence": args.mirror_stop_confidence,
            "minimum_independent_check_chars": args.mirror_stop_min_independent_check_chars,
            "maximum_pre_post_correctness_gap": args.mirror_stop_max_correctness_gap,
            "maximum_pre_post_severity_gap": args.mirror_stop_max_severity_gap,
            "required_consecutive_reliable_rounds": args.mirror_stop_consecutive_rounds,
        },
    }


def request_seed(base_seed: int, problem_idx: int, round_index: int, sample_index: int, stage: int) -> int:
    return int(base_seed + problem_idx * 100_000 + round_index * 1_000 + sample_index * 10 + stage)


def call_json(
    client: OpenAICompatibleClient,
    messages: List[Dict[str, str]],
    seed: int,
) -> Tuple[str, Dict[str, Any], float]:
    raw, latency = client.complete(messages, request_seed=seed)
    return raw, parse_json(raw, {}), latency


def sample_result_path(output_dir: Path, idx: int) -> Path:
    return output_dir / "sample_results" / f"minerva_{idx:04d}.json"


def sample_summary_path(output_dir: Path, idx: int) -> Path:
    return output_dir / "sample_summaries" / f"minerva_{idx:04d}.json"


def correctness_label(correct: Any) -> str:
    return "correct" if bool(correct) else "incorrect"


def build_compact_sample_summary(
    result: Dict[str, Any],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Dict[str, Any]:
    reference_solution = str(result.get("reference_solution", ""))
    round_results: List[Dict[str, Any]] = []
    early_stop_round: Optional[int] = None
    early_stop_reason = ""
    early_stop_stage = ""
    for round_record in result.get("rounds", []):
        candidates = round_record.get("candidates", [])
        selected_index = int(round_record.get("selected_sample_index", 1)) - 1
        selected_candidate = (
            candidates[selected_index]
            if 0 <= selected_index < len(candidates)
            else {}
        )
        pre_micro_solution = str(
            selected_candidate.get("pre_micro_candidate", {}).get("solution", "")
        )
        micro_solution = str(
            selected_candidate.get("micro_candidate", {}).get("solution", "")
        )
        selected_evaluation = round_record.get("official_evaluation", {})
        pre_micro_evaluation = evaluate_solution(
            pre_micro_solution,
            reference_solution,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        micro_evaluation = evaluate_solution(
            micro_solution,
            reference_solution,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        stop_decision = round_record.get("mirror_dynamic_stop", {})
        stopped_here = bool(stop_decision.get("should_stop"))
        if stopped_here:
            early_stop_round = int(round_record.get("round", 0))
            early_stop_reason = str(stop_decision.get("reason", ""))
            early_stop_stage = "post_micro_mirror_validation"
        round_results.append(
            {
                "round": int(round_record.get("round", 0)),
                "evolution_steps": [
                    {
                        "step": "generator_pre_micro",
                        "result": correctness_label(pre_micro_evaluation.get("correct")),
                    },
                    {
                        "step": "micro_refinement",
                        "result": correctness_label(micro_evaluation.get("correct")),
                    },
                    {
                        "step": "post_micro_guard_selected",
                        "result": correctness_label(selected_evaluation.get("correct")),
                    },
                ],
                "round_result": correctness_label(selected_evaluation.get("correct")),
                "early_stop_after_this_round": stopped_here,
            }
        )

    final_evaluation = result.get("final_official_evaluation", {})
    if early_stop_round is not None:
        termination = {
            "type": "early_stop",
            "round": early_stop_round,
            "stage": early_stop_stage,
            "reason": early_stop_reason,
        }
    elif result.get("api_failure"):
        termination = {
            "type": "api_failure",
            "round": int(result.get("api_failure", {}).get("round", result.get("final_round", 0))),
            "stage": "api_call",
            "reason": str(result.get("api_failure", {}).get("error", "")),
        }
    elif result.get("completed"):
        termination = {
            "type": "max_rounds_or_normal_completion",
            "round": int(result.get("final_round", 0)),
            "stage": "round_complete",
            "reason": "maximum_rounds_reached_or_no_early_stop",
        }
    else:
        termination = {
            "type": "in_progress",
            "round": int(result.get("final_round", 0)),
            "stage": "evolution",
            "reason": "sample_not_completed",
        }
    return {
        "idx": int(result.get("idx", -1)),
        "category": str(result.get("category", "other")),
        "completed": bool(result.get("completed")),
        "round_results": round_results,
        "termination": termination,
        "final_round": int(result.get("final_round", 0)),
        "final_result": correctness_label(final_evaluation.get("correct")),
    }


def write_compact_sample_summary(
    output_dir: Path,
    result: Dict[str, Any],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Dict[str, Any]:
    summary = build_compact_sample_summary(
        result,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    write_json(sample_summary_path(output_dir, int(result["idx"])), summary)
    return summary


def write_all_sample_outcomes(
    output_dir: Path,
    results_by_idx: Dict[int, Dict[str, Any]],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> List[Dict[str, Any]]:
    summaries = [
        build_compact_sample_summary(
            results_by_idx[idx],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for idx in sorted(results_by_idx)
    ]
    write_json(output_dir / "sample_outcomes.json", summaries)
    return summaries


def load_existing_results(output_dir: Path) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    directory = output_dir / "sample_results"
    if not directory.exists():
        return results
    for path in sorted(directory.glob("minerva_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            results[int(payload["idx"])] = payload
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return results


def write_progress_summary(
    output_dir: Path,
    results_by_idx: Dict[int, Dict[str, Any]],
    max_rounds: int,
    expected_total: int,
) -> Dict[str, Any]:
    aggregate = aggregate_sample_results(list(results_by_idx.values()), max_rounds)
    aggregate["expected_total"] = expected_total
    aggregate["remaining"] = max(0, expected_total - aggregate["completed_sample_results"])
    write_json(output_dir / "progress_summary.json", aggregate)
    write_json(output_dir / "metrics_by_round.json", aggregate["metrics_by_round"])
    return aggregate


def print_progress(aggregate: Dict[str, Any], round_index: Optional[int] = None) -> None:
    if round_index is None:
        metrics = aggregate["final_metrics"]
        completed = aggregate.get("completed_sample_results", metrics.get("overall", {}).get("total", 0))
        partial = aggregate.get("partial_sample_results", 0)
        print(
            format_metric_line(f"[progress completed-final completed={completed} partial={partial}]", metrics),
            flush=True,
        )
        return
    metrics_by_round = aggregate.get("metrics_by_round", [])
    if 1 <= round_index <= len(metrics_by_round):
        metrics = metrics_by_round[round_index - 1]
        print(format_metric_line(f"[progress selected-round={round_index}]", metrics), flush=True)


def run_sample(
    example: Dict[str, Any],
    args: argparse.Namespace,
    client: OpenAICompatibleClient,
    output_dir: Path,
    results_by_idx: Dict[int, Dict[str, Any]],
    expected_total: int,
) -> Dict[str, Any]:
    idx = int(example["idx"])
    problem = str(example["problem"])
    reference_solution = str(example["solution"])
    type_name = str(example.get("type", ""))
    category = category_for_type(type_name)
    started = time.time()
    current = {"solution": "", "final_answer": "", "rationale": "", "non_empty": False, "has_final_answer": False}
    rounds: List[Dict[str, Any]] = []
    mirror_history: List[Dict[str, Any]] = []
    role_history: List[str] = []
    role_stats = {role: {"trials": 0.0, "success": 0.0} for role in MACRO_ROLES}
    reliable_streak = 0
    api_failure: Optional[Dict[str, Any]] = None

    result: Dict[str, Any] = {
        "method": METHOD_NAME,
        "dataset": "Minerva Math",
        "split": "test",
        "idx": idx,
        "type": type_name,
        "category": category,
        "problem": problem,
        "reference_solution": reference_solution,
        "reference_answer": extract_final_answer(reference_solution),
        "completed": False,
        "rounds": rounds,
    }

    for round_index in range(1, args.max_rounds + 1):
        round_started = time.time()
        round_curriculum = select_curriculum_stage(
            round_index,
            0,
            args.samples_per_round,
            mirror_history[-1] if mirror_history else None,
        )
        macro = choose_macro_role(round_index, type_name, mirror_history, role_stats, role_history, args)
        print(
            f"[sample {idx}] round={round_index}/{args.max_rounds} "
            f"curriculum=L{round_curriculum['level']}:{round_curriculum['name']} "
            f"gate={round_curriculum['gate_reason']} macro={macro['role']}",
            flush=True,
        )
        candidates: List[Dict[str, Any]] = []
        sample_feedback = mirror_history[-1] if mirror_history else None
        connection_exit = False

        for sample_index in range(1, args.samples_per_round + 1):
            curriculum = select_curriculum_stage(
                round_index,
                sample_index - 1,
                args.samples_per_round,
                sample_feedback,
            )
            try:
                strategy_seed = request_seed(args.seed, idx, round_index, sample_index, 1)
                print(f"[sample {idx}] round={round_index} candidate={sample_index} strategy_api_start", flush=True)
                strategy_raw, _, strategy_latency = call_json(
                    client,
                    [
                        {"role": "system", "content": "Plan compact mathematical verification steps. Preserve a coherent inherited answer unless a localized, independently verified error proves it must change. Return strict JSON only."},
                        {"role": "user", "content": build_strategy_prompt(problem, type_name, macro, curriculum, current, sample_feedback)},
                    ],
                    strategy_seed,
                )
                strategy = parse_reasoning_strategy(strategy_raw, macro, curriculum)
                print(
                    f"[sample {idx}] round={round_index} candidate={sample_index} "
                    f"strategy_api_done latency={strategy_latency:.2f}s steps={len(strategy)}",
                    flush=True,
                )

                generator_seed = request_seed(args.seed, idx, round_index, sample_index, 2)
                print(f"[sample {idx}] round={round_index} candidate={sample_index} generator_api_start", flush=True)
                generator_raw, generator_payload, generator_latency = call_json(
                    client,
                    [
                        {"role": "system", "content": "You are a conservative self-evolving Minerva Math solver. Treat the inherited solution as a protected baseline and return complete strict JSON."},
                        {"role": "user", "content": build_generation_prompt(problem, type_name, round_index, args.max_rounds, macro, curriculum, current, mirror_history, strategy)},
                    ],
                    generator_seed,
                )
                pre_micro = normalize_candidate(generator_raw, generator_payload, current.get("solution", ""))
                pre_micro, pre_damper = apply_late_solution_damper(current, pre_micro, curriculum, round_index, args)
                print(
                    f"[sample {idx}] round={round_index} candidate={sample_index} "
                    f"generator_api_done latency={generator_latency:.2f}s answer={pre_micro['final_answer']!r}",
                    flush=True,
                )

                pre_mirror_seed = request_seed(args.seed, idx, round_index, sample_index, 3)
                print(f"[sample {idx}] round={round_index} candidate={sample_index} pre_mirror_api_start", flush=True)
                pre_mirror_raw, pre_mirror, pre_mirror_latency = call_json(
                    client,
                    [
                        {"role": "system", "content": "You are a pairwise mirror verifier. Prefer the inherited parent unless exact problem evidence and independent recomputation prove a localized error."},
                        {
                            "role": "user",
                            "content": build_mirror_prompt(
                                problem,
                                type_name,
                                pre_micro,
                                macro,
                                curriculum,
                                strategy,
                                parent_solution=current,
                                comparison_scope="inherited_previous_round_solution",
                            ),
                        },
                    ],
                    pre_mirror_seed,
                )
                pre_mirror = normalize_mirror_feedback(pre_mirror)
                optimized_strategy = apply_step_feedback(strategy, pre_mirror)
                print(
                    f"[sample {idx}] round={round_index} candidate={sample_index} "
                    f"pre_mirror_api_done latency={pre_mirror_latency:.2f}s",
                    flush=True,
                )

                micro_seed = request_seed(args.seed, idx, round_index, sample_index, 4)
                print(f"[sample {idx}] round={round_index} candidate={sample_index} micro_api_start", flush=True)
                micro_raw, micro_payload, micro_latency = call_json(
                    client,
                    [
                        {"role": "system", "content": "You are a conservative micro-optimizer. Make the smallest verified repair and preserve the current answer when the change gate is not satisfied."},
                        {"role": "user", "content": build_micro_prompt(problem, type_name, pre_micro, pre_mirror, macro, curriculum, optimized_strategy)},
                    ],
                    micro_seed,
                )
                micro_candidate = normalize_candidate(micro_raw, micro_payload, pre_micro["solution"])
                micro_candidate, micro_damper = apply_late_solution_damper(pre_micro, micro_candidate, curriculum, round_index, args)
                print(
                    f"[sample {idx}] round={round_index} candidate={sample_index} "
                    f"micro_api_done latency={micro_latency:.2f}s answer={micro_candidate['final_answer']!r}",
                    flush=True,
                )

                post_mirror_seed = request_seed(args.seed, idx, round_index, sample_index, 5)
                print(f"[sample {idx}] round={round_index} candidate={sample_index} post_mirror_api_start", flush=True)
                post_mirror_raw, post_mirror, post_mirror_latency = call_json(
                    client,
                    [
                        {"role": "system", "content": "You are a pairwise post-micro verifier. Reject truncation, unsupported answer changes, and external conventions that conflict with explicit problem data."},
                        {
                            "role": "user",
                            "content": build_mirror_prompt(
                                problem,
                                type_name,
                                micro_candidate,
                                macro,
                                curriculum,
                                optimized_strategy,
                                parent_solution=pre_micro,
                                comparison_scope="pre_micro_candidate",
                            ),
                        },
                    ],
                    post_mirror_seed,
                )
                post_mirror = normalize_mirror_feedback(post_mirror)
                guard = post_micro_guard_decision(
                    pre_mirror,
                    post_mirror,
                    pre_micro,
                    micro_candidate,
                    category,
                    args,
                )
                if guard["accepted"]:
                    final_candidate = micro_candidate
                    final_mirror = post_mirror
                    final_mirror_raw = post_mirror_raw
                    final_strategy = optimized_strategy
                else:
                    final_candidate = pre_micro
                    final_mirror = pre_mirror
                    final_mirror_raw = pre_mirror_raw
                    final_strategy = strategy
                parent_guard = parent_inheritance_guard_decision(
                    current,
                    final_candidate,
                    pre_mirror,
                    args,
                )
                inherited_parent = not parent_guard["candidate_accepted"]
                if inherited_parent:
                    final_candidate = dict(current)
                    final_mirror = pre_mirror
                    final_mirror_raw = pre_mirror_raw
                    final_strategy = strategy
                selection_score = mirror_selection_score(final_mirror)
                stop_decision = mirror_stop_decision(
                    final_mirror=final_mirror,
                    pre_mirror=pre_mirror,
                    post_mirror=post_mirror,
                    guard_decision=guard,
                    round_index=round_index,
                    curriculum=curriculum,
                    previous_answer=str(current.get("final_answer", "")),
                    candidate_answer=str(final_candidate.get("final_answer", "")),
                    pre_micro_answer=str(pre_micro.get("final_answer", "")),
                    micro_answer=str(micro_candidate.get("final_answer", "")),
                    inherited_parent=inherited_parent,
                    reliable_streak=reliable_streak,
                    args=args,
                )
                print(
                    f"[sample {idx}] round={round_index} candidate={sample_index} "
                    f"post_mirror_api_done latency={post_mirror_latency:.2f}s "
                    f"score={selection_score} micro_accepted={guard['accepted']}",
                    flush=True,
                )

                candidates.append(
                    {
                        "sample_index": sample_index,
                        "request_seeds": {
                            "strategy": strategy_seed,
                            "generator": generator_seed,
                            "pre_mirror": pre_mirror_seed,
                            "micro": micro_seed,
                            "post_mirror": post_mirror_seed,
                        },
                        "curriculum_stage": curriculum,
                        "strategy_raw_text": strategy_raw,
                        "strategy_latency_seconds": strategy_latency,
                        "reasoning_strategy": strategy,
                        "optimized_reasoning_strategy": final_strategy,
                        "generator_raw_text": generator_raw,
                        "generator_latency_seconds": generator_latency,
                        "pre_micro_candidate": pre_micro,
                        "pre_micro_drift_damper": pre_damper,
                        "pre_micro_mirror_raw_text": pre_mirror_raw,
                        "pre_micro_mirror_feedback": pre_mirror,
                        "pre_micro_mirror_latency_seconds": pre_mirror_latency,
                        "micro_raw_text": micro_raw,
                        "micro_latency_seconds": micro_latency,
                        "micro_candidate": micro_candidate,
                        "micro_drift_damper": micro_damper,
                        "post_micro_mirror_raw_text": post_mirror_raw,
                        "post_micro_mirror_feedback": post_mirror,
                        "post_micro_mirror_latency_seconds": post_mirror_latency,
                        "post_micro_guard_decision": guard,
                        "parent_inheritance_guard_decision": parent_guard,
                        "inherited_parent_solution": inherited_parent,
                        "selected_candidate_solution": final_candidate,
                        "mirror_raw_text": final_mirror_raw,
                        "mirror_feedback": final_mirror,
                        "mirror_selection_score": list(selection_score),
                        "selection_basis": "post_micro_mirror_feedback_only",
                        "sample_stop_decision": stop_decision,
                        "official_metrics_used_for_selection": False,
                    }
                )
                sample_feedback = final_mirror
                if stop_decision["should_stop"]:
                    break
            except (APIConnectionLimitError, RuntimeError) as exc:
                connection_exit = True
                api_failure = {
                    "round": round_index,
                    "candidate": sample_index,
                    "error": str(exc),
                    "current_solution_used_as_final": bool(current.get("solution")),
                }
                print(f"[sample {idx}] API connection limit reached: {exc}", flush=True)
                break

        if connection_exit or not candidates:
            break

        stop_ready = [item for item in candidates if item["sample_stop_decision"]["should_stop"]]
        selectable = stop_ready or candidates
        selected = max(selectable, key=lambda item: tuple(item["mirror_selection_score"]))
        previous_answer = current.get("final_answer", "")
        current = dict(selected["selected_candidate_solution"])
        selected_mirror = selected["mirror_feedback"]
        mirror_history.append(selected_mirror)
        selected_role = macro["role"]
        role_stats[selected_role]["trials"] += 1.0
        role_stats[selected_role]["success"] += clamp(
            selected_mirror.get("predicted_correctness"), 0.0, 1.0, 0.0
        )
        role_history.append(selected_role)
        stop_decision = selected["sample_stop_decision"]
        reliable_streak = int(stop_decision.get("reliable_streak_after", 0))
        official_evaluation = evaluate_solution(
            current.get("solution", ""),
            reference_solution,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        round_record = {
            "round": round_index,
            "stage": "macro_micro_evolution_with_post_micro_guard",
            "curriculum_plan": round_curriculum,
            "selected_curriculum_stage": selected["curriculum_stage"],
            "macro": macro,
            "role_stats": role_stats,
            "role_history": role_history,
            "candidates": candidates,
            "selected_sample_index": selected["sample_index"],
            "selected_mirror_selection_score": selected["mirror_selection_score"],
            "selection_score": selected["mirror_selection_score"],
            "selection_basis": (
                "post_micro_mirror_feedback_only_stop_ready_first"
                if stop_ready
                else "post_micro_mirror_feedback_only"
            ),
            "previous_final_answer": previous_answer,
            "selected_solution": current,
            "mirror_dynamic_stop": stop_decision,
            "mirror_reliable_streak": reliable_streak,
            "official_evaluation": official_evaluation,
            "official_metrics_role": "reporting_only_not_used_for_evolution",
            "round_latency_seconds": time.time() - round_started,
        }
        rounds.append(round_record)
        result.update(
            {
                "rounds": rounds,
                "final_round": round_index,
                "final_solution": current.get("solution", ""),
                "predicted_answer": current.get("final_answer", ""),
                "final_official_evaluation": official_evaluation,
                "elapsed_seconds": time.time() - started,
            }
        )
        write_json(sample_result_path(output_dir, idx), result)
        compact_summary = write_compact_sample_summary(
            output_dir,
            result,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        results_by_idx[idx] = result
        aggregate = write_progress_summary(output_dir, results_by_idx, args.max_rounds, expected_total)
        print(
            f"[sample {idx}] round={round_index} official_correct={official_evaluation['correct']} "
            f"prediction={official_evaluation['predicted_answer']!r} reference={official_evaluation['reference_answer']!r}",
            flush=True,
        )
        print(
            f"[sample {idx}] round={round_index} result={compact_summary['round_results'][-1]['round_result']} "
            f"early_stop={compact_summary['round_results'][-1]['early_stop_after_this_round']}",
            flush=True,
        )
        print_progress(aggregate, round_index)
        if stop_decision["should_stop"]:
            print(f"[sample {idx}] early_stop round={round_index} reason={stop_decision['reason']}", flush=True)
            break

    final_evaluation = evaluate_solution(
        current.get("solution", ""),
        reference_solution,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    result.update(
        {
            "completed": True,
            "final_round": rounds[-1]["round"] if rounds else 0,
            "final_solution": current.get("solution", ""),
            "predicted_answer": current.get("final_answer", ""),
            "final_official_evaluation": final_evaluation,
            "api_failure": api_failure,
            "elapsed_seconds": time.time() - started,
            "configuration": {
                "model": args.model,
                "max_output_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "max_rounds": args.max_rounds,
                "samples_per_round": args.samples_per_round,
            },
        }
    )
    write_json(sample_result_path(output_dir, idx), result)
    write_compact_sample_summary(
        output_dir,
        result,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    return result


def build_method_summary(
    args: argparse.Namespace,
    dataset_size: int,
    selected_examples: Sequence[Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "dataset": "Minerva Math",
        "split": "test",
        "dataset_file": str(Path(args.dataset).resolve()),
        "dataset_size": dataset_size,
        "selected_examples": len(selected_examples),
        "official_metric": "accuracy",
        "official_scoring_rule": "final_answer_mathematical_equivalence",
        "evaluator_capabilities": evaluator_capabilities(),
        "four_categories": list(CATEGORY_ORDER),
        "base_model": args.model,
        "base_url": args.base_url,
        "max_output_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "samples_per_round": args.samples_per_round,
        "mirror_stop": args.mirror_stop,
        "mirror_stop_protocol": {
            "minimum_rounds": None,
            "required_consecutive_reliable_rounds": args.mirror_stop_consecutive_rounds,
            "minimum_curriculum_level": None,
            "require_answer_stability": args.mirror_stop_require_answer_stability,
            "require_pre_post_agreement": args.mirror_stop_require_pre_post_agreement,
            "require_independent_answer_match": args.mirror_stop_require_independent_answer_match,
            "require_domain_verification": args.mirror_stop_require_domain_verification,
            "predicted_correctness_threshold": args.mirror_stop_correctness,
            "maximum_severity": args.mirror_stop_severity,
            "minimum_confidence": args.mirror_stop_confidence,
            "minimum_independent_check_chars": args.mirror_stop_min_independent_check_chars,
            "maximum_pre_post_correctness_gap": args.mirror_stop_max_correctness_gap,
            "maximum_pre_post_severity_gap": args.mirror_stop_max_severity_gap,
            "official_accuracy_used": False,
        },
        "result_dir": str(Path(args.output_dir).resolve()),
        "hase_v1_v2_alignment": {
            "dynamic_macro_role_selection": True,
            "minerva_category_aware_macro": True,
            "mirror_driven_easy_to_hard_curriculum": True,
            "pre_micro_mirror": True,
            "step_level_micro_optimization": True,
            "post_micro_mirror": True,
            "componentwise_post_micro_guard": True,
            "conservative_answer_change_guard": True,
            "pairwise_parent_inheritance_guard": True,
            "late_stabilizer": bool(args.enable_late_stabilizer),
            "late_solution_drift_damper": bool(args.enable_solution_drift_damper),
            "candidate_selection": "post_micro_mirror_feedback_only",
            "official_accuracy_used_for_evolution": False,
            "official_accuracy_used_for_selection": False,
            "official_accuracy_used_for_stopping": False,
            "official_metric_role": "reporting_only_after_round_selection",
        },
        **aggregate,
    }


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        "--num-samples",
        dest="limit",
        type=int,
        default=0,
        help="0 evaluates every remaining example.",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-rounds", "--rounds", dest="max_rounds", type=int, default=15)
    parser.add_argument("--samples-per-round", type=int, default=1)
    parser.add_argument("--model", default=os.getenv("HASE_MODEL", "gemini-3.1-flash-lite"))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HASE_MAX_TOKENS", "2048")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("HASE_TEMPERATURE", "0.2")))
    parser.add_argument("--base-url", default=os.getenv("HASE_BASE_URL", "https://anyaigc.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("HASE_API_KEY", ""))
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("HASE_API_TIMEOUT", "120")))
    parser.add_argument("--api-retries", type=int, default=int(os.getenv("HASE_API_RETRIES", "5")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.getenv("HASE_RETRY_SLEEP", "3")))
    parser.add_argument("--connection-failure-limit", type=int, default=int(os.getenv("HASE_CONNECTION_FAILURE_LIMIT", "20")))
    parser.add_argument("--mirror-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror-stop-consecutive-rounds", type=int, default=1)
    parser.add_argument("--mirror-stop-correctness", type=float, default=0.97)
    parser.add_argument("--mirror-stop-severity", type=float, default=0.03)
    parser.add_argument("--mirror-stop-confidence", type=float, default=0.90)
    parser.add_argument("--mirror-stop-min-independent-check-chars", type=int, default=60)
    parser.add_argument("--mirror-stop-max-correctness-gap", type=float, default=0.03)
    parser.add_argument("--mirror-stop-max-severity-gap", type=float, default=0.03)
    parser.add_argument(
        "--mirror-stop-require-answer-stability",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mirror-stop-require-pre-post-agreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mirror-stop-require-independent-answer-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mirror-stop-require-domain-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mirror-stop-blocking-modes", nargs="+", default=sorted(MATH_FAILURE_MODES))
    parser.add_argument("--post-micro-guard-margin", type=float, default=0.02)
    parser.add_argument("--post-micro-guard-correctness-margin", type=float, default=0.05)
    parser.add_argument("--post-micro-guard-severity-margin", type=float, default=0.08)
    parser.add_argument("--micro-answer-change-min-confidence", type=float, default=0.85)
    parser.add_argument("--parent-guard-confidence", type=float, default=0.80)
    parser.add_argument("--late-stability-round", type=int, default=10)
    parser.add_argument("--late-stability-correctness", type=float, default=0.60)
    parser.add_argument("--late-stability-severity", type=float, default=0.40)
    parser.add_argument("--enable-late-stabilizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-solution-drift-damper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--late-min-solution-ratio", type=float, default=0.35)
    parser.add_argument("--late-min-solution-chars", type=int, default=80)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Generate compact per-sample correct/incorrect summaries from existing detailed results without API calls.",
    )
    parser.add_argument("--summary-only", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HASE-v1 post-micro-guard-v2 on the Minerva Math test split."
    )
    add_args(parser)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(Path(args.dataset).resolve())
    examples.sort(key=lambda item: int(item.get("idx", 0)))
    selected = [item for item in examples if int(item.get("idx", 0)) >= args.start_index]
    if args.limit > 0:
        selected = selected[: args.limit]

    selected_indices = {int(item["idx"]) for item in selected}
    results_by_idx = {
        idx: result
        for idx, result in load_existing_results(output_dir).items()
        if idx in selected_indices
    }
    if args.compact_only:
        for result in results_by_idx.values():
            write_compact_sample_summary(
                output_dir,
                result,
                relative_tolerance=args.relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
            )
        summaries = write_all_sample_outcomes(
            output_dir,
            results_by_idx,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        print(
            f"Wrote {len(summaries)} compact sample summaries to "
            f"{output_dir / 'sample_summaries'} and {output_dir / 'sample_outcomes.json'}",
            flush=True,
        )
        return
    if args.summary_only:
        write_all_sample_outcomes(
            output_dir,
            results_by_idx,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        for result in results_by_idx.values():
            write_compact_sample_summary(
                output_dir,
                result,
                relative_tolerance=args.relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
            )
        aggregate = write_progress_summary(output_dir, results_by_idx, args.max_rounds, len(selected))
        summary = build_method_summary(args, len(examples), selected, aggregate)
        write_json(output_dir / "method_summary.json", summary)
        print_progress(aggregate)
        print(f"Wrote summary: {output_dir / 'method_summary.json'}", flush=True)
        return

    config = APIConfig.from_args(args)
    client = OpenAICompatibleClient(config)
    print(
        f"[start] method={METHOD_NAME} dataset={Path(args.dataset).resolve()} examples={len(selected)} "
        f"model={args.model} max_tokens={args.max_tokens} temperature={args.temperature} "
        f"seed={args.seed} max_rounds={args.max_rounds} samples_per_round={args.samples_per_round}",
        flush=True,
    )
    print(f"[output] {output_dir}", flush=True)

    for position, example in enumerate(selected, start=1):
        idx = int(example["idx"])
        existing = results_by_idx.get(idx)
        if existing and existing.get("completed") and not args.force:
            print(f"[skip] sample={idx} completed ({position}/{len(selected)})", flush=True)
            continue
        print(
            f"[sample start] idx={idx} category={category_for_type(str(example.get('type', '')))} "
            f"progress={position}/{len(selected)}",
            flush=True,
        )
        result = run_sample(example, args, client, output_dir, results_by_idx, len(selected))
        results_by_idx[idx] = result
        write_all_sample_outcomes(
            output_dir,
            results_by_idx,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        aggregate = write_progress_summary(output_dir, results_by_idx, args.max_rounds, len(selected))
        print_progress(aggregate)

    aggregate = write_progress_summary(output_dir, results_by_idx, args.max_rounds, len(selected))
    write_all_sample_outcomes(
        output_dir,
        results_by_idx,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    summary = build_method_summary(args, len(examples), selected, aggregate)
    write_json(output_dir / "method_summary.json", summary)
    print_progress(aggregate)
    print(f"Wrote method summary: {output_dir / 'method_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
