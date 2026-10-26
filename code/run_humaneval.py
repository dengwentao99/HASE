from __future__ import annotations

import argparse
import ast
import gzip
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import humaneval_evaluation_v1 as heval


class MacroRole(str, Enum):
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    STABILIZER = "stabilizer"
    CHALLENGER = "challenger"

@dataclass
class APIConfig:
    base_url: str = "https://anyaigc.com/v1"
    api_key: str = ""
    model: str = "gemini-3.1-flash-lite"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: float = 60.0
    seed: Optional[int] = None
    api_retries: int = 2
    retry_sleep: float = 2.0
    connection_failure_limit: int = 20

    @classmethod
    def from_env(cls) -> "APIConfig":
        defaults = cls()
        base_url = os.getenv("HASE_BASE_URL", defaults.base_url).rstrip("/")
        api_key = os.getenv("HASE_API_KEY", defaults.api_key)
        model = os.getenv("HASE_MODEL", defaults.model)
        temperature = float(os.getenv("HASE_TEMPERATURE", str(defaults.temperature)))
        max_tokens = int(os.getenv("HASE_MAX_TOKENS", str(defaults.max_tokens)))
        timeout = float(os.getenv("HASE_API_TIMEOUT", str(defaults.timeout)))
        seed_text = os.getenv("HASE_SEED", "" if defaults.seed is None else str(defaults.seed)).strip()
        seed = int(seed_text) if seed_text else None
        api_retries = int(os.getenv("HASE_API_RETRIES", str(defaults.api_retries)))
        retry_sleep = float(os.getenv("HASE_RETRY_SLEEP", str(defaults.retry_sleep)))
        connection_failure_limit = int(os.getenv("HASE_CONNECTION_FAILURE_LIMIT", str(defaults.connection_failure_limit)))
        if not api_key:
            raise RuntimeError(
                "Missing HASE_API_KEY. Set it before running, for example: "
                "$env:HASE_API_KEY='your_api_key'"
            )
        return cls(base_url, api_key, model, temperature, max_tokens, timeout, seed, api_retries, retry_sleep, connection_failure_limit)


class APIConnectionLimitError(RuntimeError):
    pass


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str 
    test: str  
    entry_point: str


@dataclass
class Candidate:
    task_id: str
    role: MacroRole
    round_index: int 
    prompt: str 
    raw_text: str 
    completion: str
    latency: float 

@dataclass
class EvalResult:
    task_id: str 
    passed: bool
    result: str 
    failure_mode: str
    stderr_tail: str = ""

@dataclass
class MirrorFeedback:
    raw_text: str
    parsed: Dict[str, Any] 
    latency: float


@dataclass
class EvolutionMemory:
    role_stats: Dict[MacroRole, Dict[str, float]] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    learned_rules: List[str] = field(default_factory=list)
    strategy_feedback: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for role in MacroRole:
            self.role_stats.setdefault(role, {"trials": 0.0, "success": 0.0})

    def update(self, role: MacroRole, result: EvalResult) -> None:
        stats = self.role_stats[role]
        stats["trials"] += 1.0
        stats["success"] += 1.0 if result.passed else 0.0
        if not result.passed:
            self.failure_counts[result.failure_mode] = self.failure_counts.get(result.failure_mode, 0) + 1 
            self._add_rule_from_failure(result.failure_mode)

    def success_rate(self, role: MacroRole) -> float:
        stats = self.role_stats[role] 
        if stats["trials"] == 0:
            return 0.5 
        return stats["success"] / stats["trials"] 

    def _add_rule_from_failure(self, failure_mode: str) -> None:
        rule_map = {
            "syntax_error": "Check indentation, brackets, and Python syntax; return only the function body.",
            "assertion_error": "Infer the full docstring semantics instead of only matching examples.",
            "timeout": "Avoid exponential search; prefer linear or polynomial algorithms.",
            "runtime_error": "Check empty inputs, type changes, and bounds before indexing or lookup.",
            "empty_completion": "Return executable Python function-body code, not only reasoning.",
        }
        rule = rule_map.get(failure_mode, "Re-check the prompt semantics, edge cases, and return type.")
        if rule not in self.learned_rules:
            self.learned_rules.append(rule) 

    def update_strategy_feedback(self, role: MacroRole, feedback: Dict[str, Any]) -> None:
        stats = self.role_stats[role]
        stats["trials"] += 1.0
        stats["success"] += float(feedback.get("predicted_correctness", 0.0))
        record = {"role": role.value, **feedback}
        self.strategy_feedback.append(record)
        repair_strategy = str(feedback.get("repair_strategy", "")).strip()
        if repair_strategy and repair_strategy not in self.learned_rules:
            self.learned_rules.append(repair_strategy)
        for mode in feedback.get("predicted_failure_modes", []):
            mode_text = str(mode).strip()
            if mode_text:
                self.failure_counts[mode_text] = self.failure_counts.get(mode_text, 0) + 1 


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig):
        self.config = config 

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
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
        call_started = time.time()
        for attempt in range(max_attempts):
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = response.read().decode("utf-8")
                latency = time.time() - started
                parsed = json.loads(body)
                choice = parsed["choices"][0]
                message = choice.get("message", {})
                text = extract_api_text(message.get("content"))
                if not text:
                    text = extract_api_text(message.get("reasoning_content"))
                if not text:
                    text = extract_api_text(message.get("reasoning"))
                if not text:
                    text = extract_api_text(choice.get("text"))
                return text, latency
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"API HTTP error {exc.code}: {detail}")
                if exc.code in connection_like_http_codes:
                    consecutive_connection_failures += 1
                    if consecutive_connection_failures > self.config.connection_failure_limit:
                        raise APIConnectionLimitError(
                            f"API connection failed consecutively more than {self.config.connection_failure_limit} times: {last_error}"
                        ) from exc
                else:
                    consecutive_connection_failures = 0
                if exc.code not in retryable_http_codes or (
                    attempt >= self.config.api_retries and consecutive_connection_failures <= 0
                ):
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError, http.client.RemoteDisconnected) as exc:
                last_error = RuntimeError(f"API connection error: {exc}")
                consecutive_connection_failures += 1
                if consecutive_connection_failures > self.config.connection_failure_limit:
                    raise APIConnectionLimitError(
                        f"API connection failed consecutively more than {self.config.connection_failure_limit} times: {last_error}"
                    ) from exc
                if attempt >= self.config.api_retries and consecutive_connection_failures <= 0:
                    raise last_error from exc
            if attempt + 1 < max_attempts:
                sleep_seconds = self.config.retry_sleep * (attempt + 1)
                print(
                    f"[api retry] attempt={attempt + 1}/{max_attempts - 1} "
                    f"connection_failures={consecutive_connection_failures}/{self.config.connection_failure_limit} "
                    f"sleep={sleep_seconds:.1f}s error={last_error}",
                    flush=True,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"API request failed after retries: {last_error}")

def extract_api_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part.strip())
    return str(value)


class PromptEvolver:
    ROLE_EXPLORATION_ORDER = [
        MacroRole.EXPLOITER,
        MacroRole.EXPLORER,
        MacroRole.CHALLENGER,
        MacroRole.STABILIZER,
    ] 
    CURRICULUM_STAGES = [
        {
            "level": 1,
            "name": "pass_first_minimal",
            "objective": "Write the simplest direct implementation that is most likely to pass the visible contract.",
            "constraint": "Avoid clever abstractions, broad rewrites, extra validation, and unnecessary branches.",
        },
        {
            "level": 2,
            "name": "basic_edge_aware",
            "objective": "Preserve the simple implementation while adding only the edge checks required by the prompt.",
            "constraint": "Do not change the core algorithm unless the edge case shows it is wrong.",
        },
        {
            "level": 3,
            "name": "robust_boundary",
            "objective": "Strengthen boundary behavior for empty inputs, duplicates, ordering, signs, and return type consistency.",
            "constraint": "Keep the code compact and executable; robustness must not introduce syntax or indentation risk.",
        },
        {
            "level": 4,
            "name": "adversarial_semantic",
            "objective": "Challenge ambiguous wording and likely hidden counterexamples while preserving already-correct behavior.",
            "constraint": "Only adopt semantic changes that are clearly supported by the docstring.",
        },
        {
            "level": 5,
            "name": "optimized_refinement",
            "objective": "Refine for efficiency, clarity, and generality after the mirror feedback predicts the solution is likely correct.",
            "constraint": "Never sacrifice pass-first correctness for optimization; if uncertain, fall back to the simpler version.",
        },
    ] 

    ROLE_INSTRUCTIONS = {
        MacroRole.EXPLORER: "Use an alternative but still simple algorithm. Look for hidden edge cases before coding.",
        MacroRole.EXPLOITER: "Use the most direct, idiomatic, and reliable Python implementation.",
        MacroRole.STABILIZER: "Prioritize robustness: empty inputs, duplicates, negative values, ordering, and type consistency.",
        MacroRole.CHALLENGER: "Challenge the obvious interpretation and verify the exact semantics implied by the docstring.",
    }

    ROLE_PROMPT_PROFILES = {
        MacroRole.EXPLORER: {
            "system": "Explore more than one plausible interpretation before committing to the final implementation.",
            "user": "Prefer a simple but non-obvious route when it better handles hidden cases.",
        },
        MacroRole.EXPLOITER: {
            "system": "Use the most direct reliable solution and avoid unnecessary abstraction.",
            "user": "Turn the docstring into straightforward Python with minimal moving parts.",
        },
        MacroRole.STABILIZER: {
            "system": "Treat robustness as the main objective: types, empty inputs, ordering, duplicates, and syntax.",
            "user": "Before returning code, privately stress-check edge cases and executable formatting.",
        },
        MacroRole.CHALLENGER: {
            "system": "Question the obvious reading of the task and verify the exact semantic contract.",
            "user": "Actively look for misleading examples, ambiguous wording, and off-by-one semantics.",
        },
    }

    FALLBACK_STRATEGIES = {
        MacroRole.EXPLORER: [
            ("interpretation_scan", "Compare the most likely task interpretations and keep the one best supported by the docstring."),
            ("alternative_algorithm_choice", "Choose a simple alternative algorithm only if it improves correctness on hidden cases."),
            ("implementation_commit", "Write the shortest executable function body that matches the chosen interpretation."),
            ("exploratory_check", "Check whether an unusual input shape would invalidate the chosen route."),
        ],
        MacroRole.EXPLOITER: [
            ("direct_contract_read", "Extract the input-output contract from the docstring without adding extra assumptions."),
            ("simple_algorithm_commit", "Pick the most idiomatic Python implementation with low complexity."),
            ("code_shape_check", "Ensure the answer is only an indented function body and can run immediately."),
        ],
        MacroRole.STABILIZER: [
            ("failure_repair_focus", "Start from the previous failure mode and mirror feedback, then identify the concrete repair target."),
            ("boundary_guarding", "Check empty inputs, duplicates, negative values, ordering, and type consistency before coding."),
            ("robust_implementation", "Write defensive but compact Python that preserves the required return type."),
            ("syntax_and_runtime_check", "Verify indentation, imports, variable names, and runtime behavior before final output."),
        ],
        MacroRole.CHALLENGER: [
            ("assumption_attack", "List the assumptions a naive solution would make and reject unsupported ones."),
            ("semantic_counterexample", "Use likely counterexamples to refine the intended behavior."),
            ("corrected_algorithm", "Choose an implementation that matches the refined semantics."),
            ("challenge_check", "Confirm the final code does not only satisfy examples but also the full docstring."),
        ],
    }

    def select_role_decision(
        self,
        memory: EvolutionMemory,
        last_result: Optional[EvalResult],
        round_index: int,
        last_mirror_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        used_roles: List[MacroRole] = []
        for record in memory.strategy_feedback:
            try:
                used_roles.append(MacroRole(str(record.get("role", "")).strip()))
            except ValueError:
                continue

        recommended_text = ""
        recommended_role: Optional[MacroRole] = None
        predicted_modes: List[str] = []
        confidence = 0.0
        predicted_correctness = 0.0
        if last_mirror_feedback:
            recommended_text = str(last_mirror_feedback.get("recommended_macro_role", "")).lower().strip()
            for role in self.ROLE_EXPLORATION_ORDER:
                if recommended_text == role.value:
                    recommended_role = role
                    break
            predicted_modes = [str(mode) for mode in last_mirror_feedback.get("predicted_failure_modes", [])]
            try:
                confidence = float(last_mirror_feedback.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                predicted_correctness = float(last_mirror_feedback.get("predicted_correctness", 0.0))
            except (TypeError, ValueError):
                predicted_correctness = 0.0

        total_trials = sum(memory.role_stats[role]["trials"] for role in self.ROLE_EXPLORATION_ORDER)
        recent_roles = used_roles[-3:]
        recent_role_counts = {role: recent_roles.count(role) for role in self.ROLE_EXPLORATION_ORDER}
        mode_set = set(predicted_modes)
        cold_start_prior = {
            MacroRole.EXPLOITER: 0.035,
            MacroRole.CHALLENGER: 0.020,
            MacroRole.EXPLORER: 0.015,
            MacroRole.STABILIZER: 0.0,
        }
        role_scores: Dict[MacroRole, Dict[str, float]] = {}
        for role in self.ROLE_EXPLORATION_ORDER:
            trials = memory.role_stats[role]["trials"]
            historical_success = memory.success_rate(role)
            if trials == 0:
                exploration_bonus = 0.34
            else:
                exploration_bonus = 0.18 * ((total_trials + 1.0) ** 0.5) / ((trials + 1.0) ** 0.5)
            mirror_bonus = 0.0
            if recommended_role == role:
                mirror_bonus = 0.22 + 0.18 * max(0.0, min(confidence, 1.0))
            failure_fit_bonus = 0.0
            if role == MacroRole.STABILIZER and mode_set & {"syntax_error", "runtime_error", "empty_completion"}:
                failure_fit_bonus += 0.16
            if role == MacroRole.CHALLENGER and mode_set & {"assertion_error", "edge_case_gap"}:
                failure_fit_bonus += 0.15
            if role == MacroRole.EXPLORER and mode_set & {"timeout", "edge_case_gap", "assertion_error"}:
                failure_fit_bonus += 0.12
            if role == MacroRole.EXPLOITER and last_mirror_feedback and predicted_correctness >= 0.75 and not mode_set:
                failure_fit_bonus += 0.10
            novelty_bonus = 0.10 if role not in recent_roles else 0.0
            repetition_penalty = 0.13 * recent_role_counts[role]
            if used_roles and used_roles[-1] == role:
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

        selected = max(
            self.ROLE_EXPLORATION_ORDER,
            key=lambda role: (
                role_scores[role]["score"],
                -memory.role_stats[role]["trials"],
                role.value,
            ),
        )
        return {
            "role": selected,
            "reason": (
                "Dynamic exploration scoring selected "
                f"{selected.value} from mirror feedback, historical role performance, "
                "exploration bonus, failure-mode fit, novelty bonus, and repetition penalty."
            ),
            "signals": {
                "source": "dynamic_exploration_scoring",
                "recommended_macro_role": recommended_text,
                "predicted_failure_modes": predicted_modes,
                "mirror_confidence": confidence,
                "predicted_correctness": predicted_correctness,
                "recent_roles": [role.value for role in recent_roles],
                "role_scores": {
                    role.value: {key: round(value, 6) for key, value in parts.items()}
                    for role, parts in role_scores.items()
                },
            },
        }

    def select_role(
        self,
        memory: EvolutionMemory,
        last_result: Optional[EvalResult],
        round_index: int,
        last_mirror_feedback: Optional[Dict[str, Any]] = None,
    ) -> MacroRole:
        return self.select_role_decision(memory, last_result, round_index, last_mirror_feedback)["role"]

    def select_curriculum_stage(
        self,
        round_index: int,
        sample_index: int,
        samples_per_round: int,
        last_mirror_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sample_count = max(1, samples_per_round) 
        sample_level = 1 + int(sample_index * len(self.CURRICULUM_STAGES) / sample_count) 
        round_level = min(len(self.CURRICULUM_STAGES), 1 + round_index)
        proposed_level = max(1, min(len(self.CURRICULUM_STAGES), max(sample_level, round_level)))
        predicted_correctness = 0.0
        severity_score = 0.0
        predicted_modes: List[str] = []
        if last_mirror_feedback:
            try:
                predicted_correctness = float(last_mirror_feedback.get("predicted_correctness", 0.0))
            except (TypeError, ValueError):
                predicted_correctness = 0.0
            try:
                severity_score = float(last_mirror_feedback.get("severity_score", 0.0))
            except (TypeError, ValueError):
                severity_score = 0.0
            predicted_modes = [str(mode) for mode in last_mirror_feedback.get("predicted_failure_modes", [])]

        gate_reason = "round_and_sample_progression" 
        selected_level = proposed_level
        if last_mirror_feedback and (predicted_correctness < 0.35 or severity_score >= 0.75):
            selected_level = min(proposed_level, 2)
            gate_reason = "mirror_high_risk_simplify_before_upgrade"
        elif last_mirror_feedback and predicted_correctness >= 0.75 and severity_score <= 0.25:
            selected_level = min(len(self.CURRICULUM_STAGES), proposed_level + 1) 
            gate_reason = "mirror_predicted_pass_upgrade"
        elif last_mirror_feedback:
            selected_level = min(proposed_level, 3) 
            gate_reason = "mirror_medium_risk_hold_at_robustness"

        stage = dict(self.CURRICULUM_STAGES[selected_level - 1])
        stage.update(
            {
                "sample_index": sample_index,
                "samples_per_round": samples_per_round,
                "round": round_index,
                "proposed_level": proposed_level,
                "selected_level": selected_level,
                "gate_reason": gate_reason,
                "predicted_correctness": predicted_correctness,
                "severity_score": severity_score,
                "predicted_failure_modes": predicted_modes,
                "progression_rule": "Start with pass-first minimal code; increase complexity only as mirror risk decreases or sample/round progression calls for broader stress handling.",
            }
        )
        return stage

    def build_strategy_messages(
        self,
        problem: HumanEvalProblem,
        role: MacroRole,
        selection_reason: str,
        memory: EvolutionMemory,
        last_result: Optional[EvalResult],
        last_mirror_feedback: Optional[Dict[str, Any]] = None,
        curriculum_stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        profile = self.ROLE_PROMPT_PROFILES[role] 
        learned = "\n".join(f"- {rule}" for rule in memory.learned_rules[-6:])
        mirror = format_mirror_feedback_for_prompt(last_mirror_feedback)
        system = (
            "You are the same coding agent, but this call only plans the private reasoning process. "
            "Return strict JSON only. Do not write code, markdown, prose, or tests."
        )
        user = f"""
Macro role: {role.value}
Role selection reason: {selection_reason}
Role instruction: {self.ROLE_INSTRUCTIONS[role]}
Role-specific planning focus: {profile["system"]}
Previous mirror feedback:
{mirror}

Sampling curriculum stage:
{json.dumps(curriculum_stage or {}, ensure_ascii=False, indent=2)}

Evolved reasoning rules:
{learned if learned else "- No learned rules yet."}

HumanEval prompt:
{problem.prompt}

Create the reasoning steps needed for this specific task, role, and curriculum stage.
The number of steps is not fixed; choose 2 to 7 steps only when they are useful.
Respect the curriculum: lower levels must favor pass-first simplicity; higher levels may add robustness or optimization only without breaking the simple passing solution.
Use unique snake_case step_id values that describe the actual reasoning operations.
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
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_messages(
        self,
        problem: HumanEvalProblem,
        role: MacroRole,
        selection_reason: str,
        memory: EvolutionMemory,
        last_result: Optional[EvalResult],
        reasoning_strategy: List[Dict[str, Any]],
        last_mirror_feedback: Optional[Dict[str, Any]] = None,
        curriculum_stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        profile = self.ROLE_PROMPT_PROFILES[role]
        system = (
            "You are a careful Python coding agent solving HumanEval tasks. "
            f"Current macro role: {role.value}. {profile['system']} "
            "Think privately, but output only the missing indented Python function body. "
            "Do not include markdown fences, explanations, tests, or a repeated def line."
        ) 
        learned = "\n".join(f"- {rule}" for rule in memory.learned_rules[-6:])
        mirror = format_mirror_feedback_for_prompt(last_mirror_feedback)
        user = f"""
Macro role: {role.value}
Role selection reason: {selection_reason}
Role instruction: {self.ROLE_INSTRUCTIONS[role]}
Role-specific generation directive: {profile["user"]}
Previous mirror-adversary strategy feedback:
{mirror}

Sampling curriculum stage:
{json.dumps(curriculum_stage or {}, ensure_ascii=False, indent=2)}

Current dynamic reasoning strategy generated for this round:
{json.dumps(reasoning_strategy, ensure_ascii=False, indent=2)}

Evolved reasoning rules:
{learned if learned else "- No learned rules yet."}

Mirror-adversary checklist:
- What input would break a naive implementation?
- What off-by-one, empty-input, duplicate, ordering, or type issue could make this fail?
- Can the algorithm finish within a few seconds?

HumanEval prompt:
{problem.prompt}

Privately follow the dynamic reasoning strategy and curriculum stage.
The evolution must move from simple to complex: first preserve a minimal passing solution, then add only the robustness or optimization allowed by the current curriculum level.
Return only the missing indented Python function body.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_mirror_messages(
        self,
        problem: HumanEvalProblem,
        role: MacroRole,
        selection_reason: str,
        reasoning_strategy: List[Dict[str, Any]],
        completion: str,
        memory: EvolutionMemory,
        curriculum_stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        system = (
            "You are now the mirror adversary for the same coding agent. "
            "Do not solve the task from scratch. Critique the candidate solution, "
            "predict why it may fail, and recommend the next macro role when macro selection is enabled. "
            "Return strict minified JSON only. Do not include reasoning, markdown, prose, or code fences. "
            "Your first character must be { and your last character must be }."
        )
        user = f"""
HumanEval prompt:
{problem.prompt}

Current macro role used by generator: {role.value}
Role selection reason: {selection_reason}

Current dynamic reasoning strategy used by generator:
{json.dumps(reasoning_strategy, ensure_ascii=False, indent=2)}

Current sampling curriculum stage:
{json.dumps(curriculum_stage or {}, ensure_ascii=False, indent=2)}

Candidate completion:
{completion}

Recent evolved rules:
{json.dumps(memory.learned_rules[-6:], ensure_ascii=False)}

Return only this JSON object shape, with short values:
{{
  "adversarial_summary": "short critique of the candidate",
  "predicted_failure_modes": ["syntax_error|assertion_error|timeout|runtime_error|edge_case_gap"],
  "predicted_correctness": 0.0,
  "severity_score": 0.0,
  "counterexample_ideas": ["natural-language edge cases only; do not invent hidden tests"],
  "repair_strategy": "one concise rule the generator should follow next",
  "curriculum_feedback": "whether the next sample should simplify, hold, or increase complexity",
  "step_feedback": [
    {{
      "step_id": "one of the exact step_id values from the current dynamic reasoning strategy",
      "risk": "what is weak in this generated step",
      "updated_instruction": "how this exact step should be optimized next round"
    }}
  ],
  "recommended_macro_role": "explorer|exploiter|stabilizer|challenger",
  "confidence": 0.0
}}
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_humaneval(path: Path) -> List[HumanEvalProblem]:
    opener = gzip.open if path.suffix == ".gz" else open
    problems: List[HumanEvalProblem] = []
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            problems.append(
                HumanEvalProblem(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    test=item["test"],
                    entry_point=item["entry_point"],
                )
            )
    return problems


def parse_json_payload(raw_text: str) -> Any:
    text = strip_markdown_fence(raw_text)
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)

def normalize_step_id(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return normalized or fallback


def fallback_reasoning_strategy(
    role: MacroRole,
    selection_reason: str,
    memory: EvolutionMemory,
    last_result: Optional[EvalResult],
    last_mirror_feedback: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    steps = PromptEvolver.FALLBACK_STRATEGIES[role]
    feedback_by_step = {
        str(item.get("step_id")): item
        for item in (last_mirror_feedback or {}).get("step_feedback", [])
        if isinstance(item, dict) and item.get("step_id")
    }
    strategy = []
    for index, (step_id, instruction) in enumerate(steps, start=1):
        update = feedback_by_step.get(step_id, {})
        strategy.append(
            {
                "step_index": index,
                "step_id": step_id,
                "step_name": step_id.replace("_", " "),
                "instruction": update.get("updated_instruction") or instruction,
                "base_instruction": instruction,
                "expected_private_output": "a private decision or check result used before code generation",
                "macro_direction": role.value,
                "role_focus": PromptEvolver.ROLE_INSTRUCTIONS[role],
                "selection_reason": selection_reason,
                "previous_mirror_update": update.get("updated_instruction", ""),
                "previous_mirror_risk": update.get("risk", ""),
                "last_failure_mode": None if last_result is None else last_result.failure_mode,
                "learned_rules": list(memory.learned_rules[-4:]),
                "strategy_source": "fallback",
            }
        )
    return strategy


def parse_reasoning_strategy(
    raw_text: str,
    role: MacroRole,
    selection_reason: str,
    memory: EvolutionMemory,
    last_result: Optional[EvalResult],
    last_mirror_feedback: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    try:
        parsed = parse_json_payload(raw_text)
    except json.JSONDecodeError:
        return fallback_reasoning_strategy(role, selection_reason, memory, last_result, last_mirror_feedback)
    if isinstance(parsed, dict):
        raw_steps = parsed.get("reasoning_steps") or parsed.get("steps") or []
        rationale = str(parsed.get("strategy_rationale", "")).strip()
    elif isinstance(parsed, list):
        raw_steps = parsed
        rationale = ""
    else:
        raw_steps = []
        rationale = ""
    if not isinstance(raw_steps, list):
        raw_steps = []
    strategy: List[Dict[str, Any]] = []
    used_ids = set()
    for index, item in enumerate(raw_steps[:7], start=1):
        if isinstance(item, dict):
            raw_id = str(item.get("step_id") or item.get("id") or item.get("step_name") or f"step_{index}")
            instruction = str(item.get("instruction") or item.get("description") or item.get("goal") or "").strip()
            step_name = str(item.get("step_name") or item.get("name") or raw_id).strip()
            expected = str(item.get("expected_private_output") or item.get("expected_output") or "").strip()
        else:
            raw_id = f"step_{index}"
            instruction = str(item).strip()
            step_name = raw_id
            expected = ""
        step_id = normalize_step_id(raw_id, f"step_{index}")
        if step_id in used_ids:
            step_id = f"{step_id}_{index}"
        used_ids.add(step_id)
        if not instruction:
            continue
        strategy.append(
            {
                "step_index": len(strategy) + 1,
                "step_id": step_id,
                "step_name": step_name,
                "instruction": instruction,
                "base_instruction": instruction,
                "expected_private_output": expected or "a private decision or verification result",
                "macro_direction": role.value,
                "role_focus": PromptEvolver.ROLE_INSTRUCTIONS[role],
                "selection_reason": selection_reason,
                "previous_mirror_update": "",
                "previous_mirror_risk": "",
                "last_failure_mode": None if last_result is None else last_result.failure_mode,
                "learned_rules": list(memory.learned_rules[-4:]),
                "strategy_rationale": rationale,
                "strategy_source": "model",
            }
        )
    if len(strategy) < 2:
        return fallback_reasoning_strategy(role, selection_reason, memory, last_result, last_mirror_feedback)
    return strategy


def parse_mirror_feedback(raw_text: str, valid_step_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if not raw_text.strip():
        fallback_step_id = list(valid_step_ids or ["final_self_check"])[-1]
        return {
            "adversarial_summary": "Mirror API returned empty text after repeated connection failures.",
            "recommended_macro_role": "stabilizer",
            "predicted_failure_modes": ["api_connection_failure"],
            "counterexample_ideas": [],
            "step_feedback": [
                {
                    "step_id": fallback_step_id,
                    "risk": "No mirror critique was available because the API connection failed repeatedly.",
                    "updated_instruction": "Keep the solution conservative, executable, and re-check edge cases before the next attempt.",
                }
            ],
            "repair_strategy": "Retry with a conservative, executable solution because mirror feedback was unavailable.",
            "confidence": 0.0,
            "predicted_correctness": 0.0,
            "severity_score": 1.0,
            "parse_error": True,
            "api_empty_return": True,
        }
    text = strip_markdown_fence(raw_text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
        parsed["parse_error"] = False
    except json.JSONDecodeError:
        parsed = parse_natural_language_mirror_feedback(raw_text)
        parsed["parse_error"] = True

    role = str(parsed.get("recommended_macro_role", "")).lower().strip()
    if role not in {item.value for item in PromptEvolver.ROLE_EXPLORATION_ORDER}:
        role = ""
    parsed["recommended_macro_role"] = role

    modes = parsed.get("predicted_failure_modes", [])
    if isinstance(modes, str):
        modes = [modes]
    parsed["predicted_failure_modes"] = [str(mode) for mode in modes]

    ideas = parsed.get("counterexample_ideas", [])
    if isinstance(ideas, str):
        ideas = [ideas]
    parsed["counterexample_ideas"] = [str(idea) for idea in ideas]
    parsed["repair_strategy"] = str(parsed.get("repair_strategy", "")).strip()

    step_feedback = parsed.get("step_feedback", [])
    if isinstance(step_feedback, dict):
        step_feedback = [step_feedback]
    if not isinstance(step_feedback, list):
        step_feedback = []

    normalized_steps: List[Dict[str, str]] = []
    valid_step_id_set = set(valid_step_ids or [])
    for item in step_feedback:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("step_id", "")).strip()
        if valid_step_id_set and step_id not in valid_step_id_set:
            continue
        normalized_steps.append(
            {
                "step_id": step_id,
                "risk": str(item.get("risk", "")).strip(),
                "updated_instruction": str(item.get("updated_instruction", "")).strip(),
            }
        )
    if valid_step_id_set and not normalized_steps and parsed.get("repair_strategy"):
        fallback_step_id = list(valid_step_ids or [])[-1]
        normalized_steps.append(
            {
                "step_id": fallback_step_id,
                "risk": "Mirror feedback did not target a concrete dynamic step.",
                "updated_instruction": parsed["repair_strategy"],
            }
        )
    parsed["step_feedback"] = normalized_steps

    for key in ("confidence", "predicted_correctness", "severity_score"):
        try:
            parsed[key] = float(parsed.get(key, 0.0))
        except (TypeError, ValueError):
            parsed[key] = 0.0

    modes_for_score = set(parsed.get("predicted_failure_modes", []))
    if parsed["predicted_correctness"] == 0.0:
        if not modes_for_score:
            parsed["predicted_correctness"] = 0.9
        elif modes_for_score & {"syntax_error", "empty_completion"}:
            parsed["predicted_correctness"] = 0.1
        elif modes_for_score & {"timeout", "runtime_error"}:
            parsed["predicted_correctness"] = 0.25
        elif "assertion_error" in modes_for_score:
            parsed["predicted_correctness"] = 0.4
        else:
            parsed["predicted_correctness"] = 0.6
    if parsed["severity_score"] == 0.0:
        if modes_for_score & {"syntax_error", "empty_completion"}:
            parsed["severity_score"] = 0.9
        elif modes_for_score & {"timeout", "runtime_error"}:
            parsed["severity_score"] = 0.75
        elif "assertion_error" in modes_for_score:
            parsed["severity_score"] = 0.6
        elif "edge_case_gap" in modes_for_score:
            parsed["severity_score"] = 0.35
        else:
            parsed["severity_score"] = 0.05
    return parsed


def parse_natural_language_mirror_feedback(raw_text: str) -> Dict[str, Any]:
    lowered = raw_text.lower()
    recommended_role = ""
    for role in PromptEvolver.ROLE_EXPLORATION_ORDER:
        if re.search(rf"(recommended macro role|next macro role|recommended role).*?{role.value}", lowered, re.DOTALL):
            recommended_role = role.value
            break
    predicted_modes: List[str] = []
    if re.search(r"\bsyntax_error\b|\bsyntax\b|indentation|not indented|dedent|invalid python", lowered):
        predicted_modes.append("syntax_error")
    if re.search(r"\bassertion_error\b|\bassertion\b|wrong answer|incorrect result|counterexample|fails? for", lowered):
        predicted_modes.append("assertion_error")
    if re.search(r"\btimeout\b|too slow|time limit|quadratic|exponential|complexity", lowered):
        predicted_modes.append("timeout")
    if re.search(r"\bruntime_error\b|exception|typeerror|indexerror|keyerror|zerodivision", lowered):
        predicted_modes.append("runtime_error")
    if re.search(r"\bedge_case_gap\b|edge case|boundary|empty input|duplicate|off-by-one", lowered):
        predicted_modes.append("edge_case_gap")
    if re.search(r"\bempty_completion\b|empty completion|no code|missing code", lowered):
        predicted_modes.append("empty_completion")
    predicted_modes = list(dict.fromkeys(predicted_modes)) 
    repair_match = re.search(
        r"(repair strategy|strategy|fix)\s*:\s*(.*?)(?:\n|$)",
        raw_text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"(repair strategy|strategy|fix)\s*：\s*(.*?)(?:\n|$)",
        raw_text,
        flags=re.IGNORECASE,
    )
    repair_strategy = repair_match.group(2).strip() if repair_match else ""
    if not repair_strategy and "natural language" in lowered and "code" in lowered:
        repair_strategy = "Return only executable Python function-body code, not prose."
    summary = raw_text.strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "...[truncated]"
    return {
        "adversarial_summary": summary,
        "recommended_macro_role": recommended_role,
        "predicted_failure_modes": predicted_modes,
        "counterexample_ideas": [],
        "step_feedback": natural_language_step_feedback(lowered, repair_strategy),
        "repair_strategy": repair_strategy,
        "confidence": 0.3 if predicted_modes or repair_strategy else 0.0,
    }


def natural_language_step_feedback(lowered_text: str, repair_strategy: str) -> List[Dict[str, str]]:
    feedback = []
    if "syntax" in lowered_text or "indent" in lowered_text or "code" in lowered_text:
        feedback.append(
            {
                "step_id": "final_self_check",
                "risk": "The generated answer may contain prose, malformed indentation, or invalid Python syntax.",
                "updated_instruction": repair_strategy
                or "Before answering, verify that the output is only executable indented Python function-body code.",
            }
        ) 
    if "edge" in lowered_text or "empty" in lowered_text or "duplicate" in lowered_text:
        feedback.append(
            {
                "step_id": "find_edge_cases",
                "risk": "The solution may miss boundary inputs such as empty values, duplicates, or unusual ordering.",
                "updated_instruction": "Explicitly check boundary cases before choosing the implementation.",
            }
        )
    if "algorithm" in lowered_text or "complexity" in lowered_text or "timeout" in lowered_text:
        feedback.append(
            {
                "step_id": "design_algorithm",
                "risk": "The selected algorithm may be too slow or mismatched to the task constraints.",
                "updated_instruction": "Select the simplest algorithm with acceptable complexity before coding.",
            }
        )
    return feedback


def format_mirror_feedback_for_prompt(feedback: Optional[Dict[str, Any]]) -> str:
    if not feedback:
        return "- No previous mirror feedback yet."
    lines = [
        f"- adversarial_summary: {feedback.get('adversarial_summary', '')}",
        f"- predicted_failure_modes: {feedback.get('predicted_failure_modes', [])}",
        f"- counterexample_ideas: {feedback.get('counterexample_ideas', [])}",
        f"- repair_strategy: {feedback.get('repair_strategy', '')}",
        f"- step_feedback: {feedback.get('step_feedback', [])}",
        f"- recommended_macro_role: {feedback.get('recommended_macro_role', '')}",
    ]
    return "\n".join(lines)


def apply_step_feedback(
    reasoning_strategy: List[Dict[str, Any]],
    mirror_feedback: Dict[str, Any],
) -> List[Dict[str, Any]]:
    feedback_by_step = {
        item["step_id"]: item
        for item in mirror_feedback.get("step_feedback", [])
        if isinstance(item, dict) and item.get("step_id")
    }
    updated_strategy = []
    for step in reasoning_strategy:
        updated = dict(step)
        step_feedback = feedback_by_step.get(step["step_id"])
        base_instruction = step.get("instruction") or step.get("base_instruction", "")
        if step_feedback:
            updated["mirror_optimized"] = True
            updated["mirror_risk"] = step_feedback.get("risk", "")
            updated["next_instruction"] = step_feedback.get("updated_instruction", "") or base_instruction
            updated["instruction"] = updated["next_instruction"]
        else:
            updated["mirror_optimized"] = False
            updated["mirror_risk"] = ""
            updated["next_instruction"] = step.get("previous_mirror_update", "") or base_instruction
        updated_strategy.append(updated) 
    return updated_strategy


def mirror_selection_score(mirror_feedback: Dict[str, Any]) -> Tuple[float, float, int]:
    predicted_correctness = float(mirror_feedback.get("predicted_correctness", 0.0))
    severity_score = float(mirror_feedback.get("severity_score", 0.0))
    confidence = float(mirror_feedback.get("confidence", 0.0))
    failure_count = len(mirror_feedback.get("predicted_failure_modes", []))
    return (predicted_correctness, -severity_score, confidence - 0.05 * failure_count)


def mirror_reliability_stop_decision(
    mirror_feedback: Dict[str, Any],
    round_index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    predicted_correctness = float(mirror_feedback.get("predicted_correctness", 0.0))
    severity_score = float(mirror_feedback.get("severity_score", 0.0))
    confidence = float(mirror_feedback.get("confidence", 0.0))
    predicted_modes = [str(mode) for mode in mirror_feedback.get("predicted_failure_modes", [])]
    blocking_modes = set(predicted_modes) & set(args.mirror_stop_blocking_modes)
    enough_rounds = round_index + 1 >= args.mirror_stop_min_rounds
    reliable = (
        args.mirror_stop
        and enough_rounds
        and predicted_correctness >= args.mirror_stop_correctness
        and severity_score <= args.mirror_stop_severity
        and confidence >= args.mirror_stop_confidence
        and not blocking_modes
    )
    if not args.mirror_stop:
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
        reason = "mirror_feedback_predicts_current_result_is_reliable"
    return {
        "should_stop": reliable,
        "reason": reason,
        "predicted_correctness": predicted_correctness,
        "severity_score": severity_score,
        "confidence": confidence,
        "predicted_failure_modes": predicted_modes,
        "blocking_failure_modes": sorted(blocking_modes),
        "thresholds": {
            "mirror_stop": args.mirror_stop,
            "mirror_stop_min_rounds": args.mirror_stop_min_rounds,
            "mirror_stop_correctness": args.mirror_stop_correctness,
            "mirror_stop_severity": args.mirror_stop_severity,
            "mirror_stop_confidence": args.mirror_stop_confidence,
            "mirror_stop_blocking_modes": list(args.mirror_stop_blocking_modes),
        },
    }


def strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def extract_body_from_full_function(code: str, entry_point: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            body_nodes = list(node.body)
            if body_nodes and isinstance(body_nodes[0], ast.Expr) and isinstance(body_nodes[0].value, ast.Constant):
                if isinstance(body_nodes[0].value.value, str):
                    body_nodes = body_nodes[1:]
            segments = [ast.get_source_segment(code, body_node) for body_node in body_nodes]
            segments = [segment for segment in segments if segment]
            if not segments:
                return "    pass\n"
            return "\n".join(indent_as_body(segment) for segment in segments) + "\n"
    return None


def indent_as_body(code: str) -> str:
    lines = code.strip("\n").splitlines()
    return "\n".join(line if line.startswith("    ") else f"    {line}" for line in lines)


def normalize_completion(raw_text: str, entry_point: str) -> str:
    code = strip_markdown_fence(raw_text)
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL | re.IGNORECASE).strip()
    extracted = extract_body_from_full_function(code, entry_point)
    if extracted is not None:
        return extracted
    code = re.sub(r"^```.*?$|```$", "", code, flags=re.MULTILINE).strip()
    code = re.sub(r"^Here is.*?:", "", code, flags=re.IGNORECASE).strip()
    if not code:
        return ""
    return indent_as_body(code) + "\n"


def evaluate_completion(
    problem: HumanEvalProblem,
    completion: str,
    timeout: float,
    python_executable: str,
) -> EvalResult:
    if not completion.strip():
        return EvalResult(problem.task_id, False, "empty completion", "empty_completion")  
    check_program = (
        problem.prompt
        + completion
        + "\n"
        + problem.test
        + "\n"
        + f"check({problem.entry_point})\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_file = Path(tmpdir) / "candidate.py"
        candidate_file.write_text(check_program, encoding="utf-8")
        try:
            result = subprocess.run(
                [python_executable, str(candidate_file)],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(problem.task_id, False, "timed out", "timeout")
    if result.returncode == 0:
        return EvalResult(problem.task_id, True, "passed", "passed")
    stderr = result.stderr[-1200:] if result.stderr else ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        mode = "syntax_error"
    elif "AssertionError" in stderr:
        mode = "assertion_error"
    else:
        mode = "runtime_error"
    return EvalResult(problem.task_id, False, mode, mode, stderr)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    product = 1.0
    for value in range(n - c + 1, n + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def compute_pass_metrics(results_by_task: Dict[str, List[EvalResult]], ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        scores: List[float] = []
        for task_results in results_by_task.values():
            n = len(task_results)
            c = sum(1 for result in task_results if result.passed)
            if n >= k:
                scores.append(estimate_pass_at_k(n, c, k))
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)
    return metrics


def compute_metrics_by_round(
    results_by_round: Dict[int, Dict[str, List[EvalResult]]],
    ks: Sequence[int],
    max_rounds: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    metrics_by_round: Dict[int, Dict[str, Any]] = {}
    if max_rounds is None:
        target_rounds = sorted(results_by_round)
    else:
        target_rounds = list(range(max_rounds))
    task_ids = sorted(
        {
            task_id
            for round_results in results_by_round.values()
            for task_id in round_results
        }
    )
    task_rounds: Dict[str, List[int]] = {
        task_id: sorted(
            round_index
            for round_index, round_results in results_by_round.items()
            if task_id in round_results
        )
        for task_id in task_ids
    }
    for round_index in target_rounds:
        round_results: Dict[str, List[EvalResult]] = {}
        actual_tasks = 0
        carried_forward_tasks = 0
        carry_forward_sources: Dict[str, int] = {}
        for task_id in task_ids:
            available_rounds = task_rounds.get(task_id, [])
            if round_index in available_rounds:
                source_round = round_index
                actual_tasks += 1
            else:
                previous_rounds = [value for value in available_rounds if value < round_index]
                if not previous_rounds:
                    continue
                source_round = previous_rounds[-1]
                carried_forward_tasks += 1
                carry_forward_sources[task_id] = source_round
            round_results[task_id] = results_by_round[source_round][task_id]
        pass_metrics = heval.compute_pass_metrics(round_results, ks)
        attempts = sum(len(task_results) for task_results in round_results.values())
        passed = sum(1 for task_results in round_results.values() for result in task_results if result.passed)
        metrics_by_round[round_index + 1] = {
            "round": round_index,
            "evolution_iteration": round_index + 1,
            "reported_round": round_index + 1,
            "num_tasks": len(round_results),
            "actual_tasks": actual_tasks,
            "carried_forward_tasks": carried_forward_tasks,
            "carry_forward_sources": carry_forward_sources,
            "attempts": attempts,
            "passed": passed,
            "pass_rate": passed / attempts if attempts else 0.0,
            **pass_metrics,
        }
    return metrics_by_round


def compute_overall_solved_rate(results_by_task: Dict[str, List[EvalResult]]) -> float:
    if not results_by_task:
        return 0.0
    solved = sum(1 for task_results in results_by_task.values() if any(result.passed for result in task_results))
    return solved / len(results_by_task)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_result_from_hase_row(row: Dict[str, Any]) -> heval.EvalResult:
    return heval.EvalResult(
        task_id=row["task_id"],
        passed=bool(row["passed"]),
        result=str(row["result"]),
        failure_mode=str(row["failure_mode"]),
        stderr_tail=str(row.get("stderr_tail", "")),
        execution_time=float(row.get("candidate_execution_time", row.get("execution_time", 0.0))),
    )


def hase_sample_result_dir(args: argparse.Namespace, output_prefix: Path) -> Path:
    configured = getattr(args, "sample_result_dir", None)
    return Path(configured) if configured else output_prefix.parent / f"{output_prefix.name}.sample_results"


def hase_sample_result_path(directory: Path, task_id: str) -> Path:
    return directory / safe_task_filename(task_id)


def write_hase_sample_result(
    path: Path,
    problem: heval.HumanEvalProblem,
    sample_index: int,
    sample_rows: Sequence[Dict[str, Any]],
    result_rows: Sequence[Dict[str, Any]],
    strategy_feedback_added: Sequence[Dict[str, Any]],
    trace: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "HASE",
        "task_id": problem.task_id,
        "sample_index": sample_index,
        "entry_point": problem.entry_point,
        "completed": True,
        "sample_rows": list(sample_rows),
        "result_rows": list(result_rows),
        "strategy_feedback_added": list(strategy_feedback_added),
        "trace_summary": trace.get("summary", {}),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_hase_sample_result(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_hase_strategy_feedback(memory: EvolutionMemory, rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        role_text = str(row.get("role", "")).strip()
        try:
            role = MacroRole(role_text)
        except ValueError:
            continue
        memory.update_strategy_feedback(role, row)


def memory_snapshot(memory: EvolutionMemory) -> Dict[str, Any]:
    return {
        "role_stats": {role.value: dict(stats) for role, stats in memory.role_stats.items()},
        "failure_counts": dict(memory.failure_counts),
        "learned_rules": list(memory.learned_rules),
        "recent_strategy_feedback": list(memory.strategy_feedback[-5:]),
    }

def safe_task_filename(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id) + ".json"


def write_task_trace(trace_dir: Optional[Path], trace: Dict[str, Any]) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / safe_task_filename(trace["task_id"])
    trace_file.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8") 

def start_timing() -> Dict[str, float]:
    now = time.perf_counter()
    return {"started_at": now} 


def finish_timing(timing: Dict[str, float]) -> Dict[str, float]:
    ended = time.perf_counter()
    started = timing["started_at"]
    return {
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": ended - started,
    }


def role_selection_reason(
    last_result: Optional[EvalResult],
    round_index: int,
    last_mirror_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    if round_index == 0:
        return "first_round_baseline_exploitation"
    if last_mirror_feedback and last_mirror_feedback.get("recommended_macro_role"):
        return f"mirror_recommended_macro_role:{last_mirror_feedback.get('recommended_macro_role')}"
    if last_result is None:
        return "no_previous_result"
    return f"previous_failure_mode:{last_result.failure_mode}"

def run_hase_humaneval(args: argparse.Namespace) -> Dict[str, Any]:
    config = APIConfig.from_env()
    if args.seed is not None:
        config.seed = args.seed
    if args.api_retries is not None:
        config.api_retries = args.api_retries
    if args.retry_sleep is not None:
        config.retry_sleep = args.retry_sleep 
    if args.connection_failure_limit is not None:
        config.connection_failure_limit = args.connection_failure_limit
    client = OpenAICompatibleClient(config)
    evolver = PromptEvolver()
    memory = EvolutionMemory()
    problems = heval.read_humaneval(Path(args.problem_file))
    if args.limit is not None:
        problems = problems[: args.limit] 

    output_prefix = Path(args.output_prefix) 
    sample_file = output_prefix.with_suffix(".samples.jsonl")
    result_file = output_prefix.with_suffix(".results.jsonl")
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    per_sample_dir = hase_sample_result_dir(args, output_prefix)
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file.unlink(missing_ok=True) 
    result_file.unlink(missing_ok=True)
    results_by_task: Dict[str, List[EvalResult]] = {}
    results_by_round: Dict[int, Dict[str, List[EvalResult]]] = {}
    skipped_tasks: List[str] = []
    completed_sample_files: Dict[str, str] = {}

    for index, problem in enumerate(problems, start=1):
        sample_result_path = hase_sample_result_path(per_sample_dir, problem.task_id)
        if sample_result_path.exists():
            payload = load_hase_sample_result(sample_result_path)
            print(f"[{index}/{len(problems)}] {problem.task_id} skip existing {sample_result_path}", flush=True)
            task_results = [eval_result_from_hase_row(row) for row in payload.get("result_rows", [])]
            results_by_task[problem.task_id] = task_results
            for row in payload.get("sample_rows", []):
                append_jsonl(sample_file, row)
            for row in payload.get("result_rows", []):
                append_jsonl(result_file, row)
                result = eval_result_from_hase_row(row)
                round_bucket = results_by_round.setdefault(int(row.get("round", 0)), {})
                round_bucket.setdefault(problem.task_id, []).append(result)
            replay_hase_strategy_feedback(memory, payload.get("strategy_feedback_added", []))
            skipped_tasks.append(problem.task_id)
            completed_sample_files[problem.task_id] = str(sample_result_path)
            continue
        task_timing = start_timing()
        last_result: Optional[EvalResult] = None
        last_mirror_feedback: Optional[Dict[str, Any]] = None
        task_results: List[EvalResult] = []
        task_sample_rows: List[Dict[str, Any]] = []
        task_result_rows: List[Dict[str, Any]] = []
        task_strategy_feedback_start = len(memory.strategy_feedback)
        dataset_timing = start_timing()
        trace: Dict[str, Any] = {
            "task_id": problem.task_id,
            "sample_index": index - 1,
            "dataset": "HumanEval",
            "model": config.model,
            "base_url": config.base_url,
            "problem": {
                "stage": "dataset_loading",
                "phase": "dataset",
                "task_id": problem.task_id,
                "entry_point": problem.entry_point,
                "prompt": problem.prompt,
                "test_included_in_trace": False,
                "timing": finish_timing(dataset_timing),
            },
            "stages": [],
            "summary": {},
        }
        write_task_trace(trace_dir, trace) 
        print(f"[{index}/{len(problems)}] {problem.task_id}", flush=True)
        mirror_stop_record: Optional[Dict[str, Any]] = None
        api_connection_exit_record: Optional[Dict[str, Any]] = None
        for round_index in range(args.rounds):
            macro_timing = start_timing()
            memory_before_role = memory_snapshot(memory)
            role_decision = evolver.select_role_decision(memory, last_result, round_index, last_mirror_feedback)
            role = role_decision["role"]
            selection_reason = role_decision["reason"]
            macro_timing_result = finish_timing(macro_timing)
            trace["stages"].append(
                {
                    "stage": "macro_role_selection",
                    "phase": "macro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "selected_role": role.value,
                    "selection_reason": selection_reason,
                    "selection_signals": role_decision.get("signals", {}),
                    "last_result": None
                    if last_result is None
                    else {
                        "passed": last_result.passed,
                        "result": last_result.result,
                        "failure_mode": last_result.failure_mode,
                    },
                    "last_mirror_feedback": last_mirror_feedback,
                    "memory_before": memory_before_role,
                    "role_instruction": evolver.ROLE_INSTRUCTIONS[role],
                    "timing": macro_timing_result,
                }
            )
            write_task_trace(trace_dir, trace)
            round_curriculum_stage = evolver.select_curriculum_stage(
                round_index,
                0,
                args.samples_per_round,
                last_mirror_feedback,
            )
            trace["stages"].append(
                {
                    "stage": "curriculum_stage_selection",
                    "phase": "curriculum",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "sample": None,
                    "scope": "round_strategy_planning",
                    "selected_role": role.value,
                    "curriculum_stage": round_curriculum_stage,
                }
            )
            write_task_trace(trace_dir, trace)
            strategy_prompt_timing = start_timing()
            strategy_messages = evolver.build_strategy_messages(
                problem,
                role,
                selection_reason,
                memory,
                last_result,
                last_mirror_feedback,
                round_curriculum_stage,
            )
            strategy_prompt_timing_result = finish_timing(strategy_prompt_timing)
            trace["stages"].append(
                {
                    "stage": "reasoning_strategy_prompt",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "selected_role": role.value,
                    "selection_reason": selection_reason,
                    "curriculum_stage": round_curriculum_stage,
                    "system_prompt": strategy_messages[0]["content"],
                    "user_prompt": strategy_messages[1]["content"],
                    "timing": strategy_prompt_timing_result,
                }
            )
            write_task_trace(trace_dir, trace)
            strategy_api_timing = start_timing()
            try:
                strategy_raw_text, strategy_latency = client.complete(strategy_messages)
            except APIConnectionLimitError as exc:
                strategy_api_timing_result = finish_timing(strategy_api_timing)
                empty_completion = ""
                result = heval.evaluate_completion(
                    problem,
                    empty_completion,
                    timeout=args.eval_timeout,
                    python_executable=args.python,
                )
                candidate = Candidate(
                    task_id=problem.task_id,
                    role=role,
                    round_index=round_index,
                    prompt=strategy_messages[-1]["content"],
                    raw_text="",
                    completion=empty_completion,
                    latency=0.0,
                )
                task_results.append(result)
                round_bucket = results_by_round.setdefault(round_index, {})
                round_bucket.setdefault(problem.task_id, []).append(result)
                sample_row = {
                    "task_id": candidate.task_id,
                    "completion": candidate.completion,
                    "role": candidate.role.value,
                    "round": candidate.round_index,
                    "sample": 0,
                    "curriculum_stage": round_curriculum_stage,
                    "latency": candidate.latency,
                    "api_connection_limit_exit": True,
                    "api_failure_stage": "reasoning_strategy_generation",
                }
                result_row = {
                    "task_id": result.task_id,
                    "passed": result.passed,
                    "result": result.result,
                    "failure_mode": result.failure_mode,
                    "role": role.value,
                    "round": round_index,
                    "sample": 0,
                    "curriculum_stage": round_curriculum_stage,
                    "stderr_tail": result.stderr_tail,
                    "candidate_execution_time": getattr(result, "execution_time", 0.0),
                    "api_connection_limit_exit": True,
                    "api_failure_stage": "reasoning_strategy_generation",
                }
                task_sample_rows.append(sample_row)
                task_result_rows.append(result_row)
                append_jsonl(sample_file, sample_row)
                append_jsonl(result_file, result_row)
                api_connection_exit_record = {
                    "stopped": True,
                    "round": round_index,
                    "evolution_iteration": round_index + 1,
                    "sample": 0,
                    "stage": "reasoning_strategy_generation",
                    "used_current_round_result": False,
                    "returned_empty_completion": True,
                    "reason": str(exc),
                }
                trace["stages"].append(
                    {
                        "stage": "api_connection_limit_exit",
                        "phase": "control",
                        "evolution_iteration": round_index + 1,
                        "round": round_index,
                        "sample": 0,
                        "api_failure_stage": "reasoning_strategy_generation",
                        "used_current_round_result": False,
                        "returned_empty_completion": True,
                        "error": str(exc),
                        "timing": strategy_api_timing_result,
                    }
                )
                write_task_trace(trace_dir, trace)
                print(f"  stop=api_connection_limit stage=reasoning_strategy_generation round={round_index}", flush=True)
                break
            reasoning_strategy = parse_reasoning_strategy(
                strategy_raw_text,
                role,
                selection_reason,
                memory,
                last_result,
                last_mirror_feedback,
            )
            strategy_api_timing_result = finish_timing(strategy_api_timing)
            trace["stages"].append(
                {
                    "stage": "reasoning_strategy_generation",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "selected_role": role.value,
                    "selection_reason": selection_reason,
                    "model": config.model,
                    "api_latency": strategy_latency,
                    "timing": strategy_api_timing_result,
                    "raw_text": strategy_raw_text,
                    "reasoning_strategy": reasoning_strategy,
                    "num_reasoning_steps": len(reasoning_strategy),
                }
            )
            write_task_trace(trace_dir, trace)
            round_sample_records: List[Dict[str, Any]] = []
            sample_feedback_for_curriculum = last_mirror_feedback
            stop_after_current_sample = False
            for sample_index in range(args.samples_per_round):
                sample_curriculum_stage = evolver.select_curriculum_stage(
                    round_index,
                    sample_index,
                    args.samples_per_round,
                    sample_feedback_for_curriculum,
                )
                prompt_timing = start_timing()
                messages = evolver.build_messages(
                    problem,
                    role,
                    selection_reason,
                    memory,
                    last_result,
                    reasoning_strategy,
                    sample_feedback_for_curriculum,
                    sample_curriculum_stage,
                )
                prompt_timing_result = finish_timing(prompt_timing)
                trace["stages"].append(
                {
                    "stage": "prompt_construction",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "system_prompt": messages[0]["content"],
                        "user_prompt": messages[1]["content"],
                        "learned_rules_used": list(memory.learned_rules[-6:]),
                        "timing": prompt_timing_result,
                    }
                )
                write_task_trace(trace_dir, trace)
                api_timing = start_timing()
                generation_api_error: Optional[str] = None
                try:
                    raw_text, latency = client.complete(messages)
                except APIConnectionLimitError as exc:
                    raw_text, latency = "", 0.0
                    generation_api_error = str(exc)
                api_timing_result = finish_timing(api_timing)
                trace["stages"].append(
                {
                    "stage": "api_generation",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "sample": sample_index,
                        "model": config.model,
                        "api_latency": latency,
                        "timing": api_timing_result,
                        "raw_text": raw_text,
                    }
                )
                write_task_trace(trace_dir, trace) 
                normalization_timing = start_timing()
                completion = heval.normalize_completion(raw_text, problem.entry_point)
                normalization_timing_result = finish_timing(normalization_timing)
                trace["stages"].append(
                {
                    "stage": "completion_normalization",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                        "sample": sample_index,
                        "entry_point": problem.entry_point,
                        "completion": completion,
                        "timing": normalization_timing_result,
                    }
                )
                write_task_trace(trace_dir, trace)
                candidate = Candidate(
                    task_id=problem.task_id,
                    role=role,
                    round_index=round_index,
                    prompt=messages[-1]["content"],
                    raw_text=raw_text,
                    completion=completion,
                    latency=latency,
                )
                evaluation_timing = start_timing()
                result = heval.evaluate_completion(
                    problem,
                    candidate.completion,
                    timeout=args.eval_timeout,
                    python_executable=args.python,
                )
                evaluation_timing_result = finish_timing(evaluation_timing)
                trace["stages"].append(
                {
                    "stage": "local_humaneval_evaluation",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "sample": sample_index,
                        "passed": result.passed,
                        "result": result.result,
                        "failure_mode": result.failure_mode,
                        "stderr_tail": result.stderr_tail,
                        "eval_timeout": args.eval_timeout,
                        "candidate_execution_time": getattr(result, "execution_time", 0.0),
                        "timing": evaluation_timing_result,
                    }
                )
                if generation_api_error:
                    task_results.append(result)
                    round_bucket = results_by_round.setdefault(round_index, {})
                    round_bucket.setdefault(problem.task_id, []).append(result)
                    sample_row = {
                        "task_id": candidate.task_id,
                        "completion": candidate.completion,
                        "role": candidate.role.value,
                        "round": candidate.round_index,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "latency": candidate.latency,
                        "api_connection_limit_exit": True,
                        "api_failure_stage": "api_generation",
                    }
                    result_row = {
                        "task_id": result.task_id,
                        "passed": result.passed,
                        "result": result.result,
                        "failure_mode": result.failure_mode,
                        "role": role.value,
                        "round": round_index,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "stderr_tail": result.stderr_tail,
                        "candidate_execution_time": getattr(result, "execution_time", 0.0),
                        "api_connection_limit_exit": True,
                        "api_failure_stage": "api_generation",
                    }
                    task_sample_rows.append(sample_row)
                    task_result_rows.append(result_row)
                    append_jsonl(sample_file, sample_row)
                    append_jsonl(result_file, result_row)
                    api_connection_exit_record = {
                        "stopped": True,
                        "round": round_index,
                        "evolution_iteration": round_index + 1,
                        "sample": sample_index,
                        "stage": "api_generation",
                        "used_current_round_result": False,
                        "returned_empty_completion": True,
                        "reason": generation_api_error,
                    }
                    trace["stages"].append(
                        {
                            "stage": "api_connection_limit_exit",
                            "phase": "control",
                            "evolution_iteration": round_index + 1,
                            "round": round_index,
                            "sample": sample_index,
                            "curriculum_stage": sample_curriculum_stage,
                            "api_failure_stage": "api_generation",
                            "used_current_round_result": False,
                            "returned_empty_completion": True,
                            "error": generation_api_error,
                        }
                    )
                    write_task_trace(trace_dir, trace)
                    print(f"  stop=api_connection_limit stage=api_generation round={round_index} sample={sample_index}", flush=True)
                    break
                mirror_prompt_timing = start_timing()
                mirror_messages = evolver.build_mirror_messages(
                    problem,
                    role,
                    selection_reason,
                    reasoning_strategy,
                    candidate.completion,
                    memory,
                    sample_curriculum_stage,
                )
                mirror_prompt_timing_result = finish_timing(mirror_prompt_timing)
                trace["stages"].append(
                {
                    "stage": "mirror_adversary_prompt",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "system_prompt": mirror_messages[0]["content"],
                        "user_prompt": mirror_messages[1]["content"],
                        "timing": mirror_prompt_timing_result,
                    }
                )
                write_task_trace(trace_dir, trace)
                mirror_feedback_timing = start_timing()
                mirror_api_error: Optional[str] = None
                try:
                    mirror_raw_text, mirror_latency = client.complete(mirror_messages) 
                except APIConnectionLimitError as exc:
                    mirror_raw_text, mirror_latency = "", 0.0
                    mirror_api_error = str(exc)
                mirror_parsed = parse_mirror_feedback(
                    mirror_raw_text,
                    valid_step_ids=[step["step_id"] for step in reasoning_strategy],
                )
                sample_feedback_for_curriculum = mirror_parsed 
                mirror_feedback_timing_result = finish_timing(mirror_feedback_timing)
                strategy_update_timing = start_timing()
                optimized_strategy = apply_step_feedback(reasoning_strategy, mirror_parsed)
                strategy_update_timing_result = finish_timing(strategy_update_timing)
                trace["stages"].append(
                {
                    "stage": "mirror_adversary_feedback",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "model": config.model,
                        "api_latency": mirror_latency,
                        "timing": mirror_feedback_timing_result,
                        "raw_text": mirror_raw_text,
                        "parsed_feedback": mirror_parsed,
                    }
                )
                trace["stages"].append(
                {
                    "stage": "reasoning_strategy_step_update",
                    "phase": "micro",
                    "evolution_iteration": round_index + 1,
                    "round": round_index,
                    "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "original_strategy": reasoning_strategy,
                        "mirror_step_feedback": mirror_parsed.get("step_feedback", []),
                        "optimized_strategy_for_next_iteration": optimized_strategy,
                        "timing": strategy_update_timing_result,
                    }
                )
                task_results.append(result)
                round_bucket = results_by_round.setdefault(round_index, {})
                round_bucket.setdefault(problem.task_id, []).append(result)
                round_sample_records.append(
                    {
                        "candidate": candidate,
                        "result": result,
                        "mirror_feedback": mirror_parsed,
                        "optimized_strategy": optimized_strategy,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "mirror_selection_score": mirror_selection_score(mirror_parsed),
                    }
                )
                sample_row = {
                    "task_id": candidate.task_id,
                    "completion": candidate.completion,
                    "role": candidate.role.value,
                    "round": candidate.round_index,
                    "sample": sample_index,
                    "curriculum_stage": sample_curriculum_stage,
                    "latency": candidate.latency,
                }
                result_row = {
                    "task_id": result.task_id,
                    "passed": result.passed,
                    "result": result.result,
                    "failure_mode": result.failure_mode,
                    "role": role.value,
                    "round": round_index,
                    "sample": sample_index,
                    "curriculum_stage": sample_curriculum_stage,
                    "stderr_tail": result.stderr_tail,
                    "candidate_execution_time": getattr(result, "execution_time", 0.0),
                }
                task_sample_rows.append(sample_row)
                task_result_rows.append(result_row)
                append_jsonl(sample_file, sample_row)
                append_jsonl(result_file, result_row)
                print(
                    f"  round={round_index} sample={sample_index} "
                    f"role={role.value} curriculum=L{sample_curriculum_stage['level']}:{sample_curriculum_stage['name']} result={result.result}",
                    flush=True,
                )
                sample_stop_decision = mirror_reliability_stop_decision(mirror_parsed, round_index, args)
                round_sample_records[-1]["sample_stop_decision"] = sample_stop_decision
                trace["stages"].append(
                    {
                        "stage": "mirror_reliability_sample_stop_check",
                        "phase": "control",
                        "evolution_iteration": round_index + 1,
                        "round": round_index,
                        "sample": sample_index,
                        "curriculum_stage": sample_curriculum_stage,
                        "selection_basis": "current_sample_mirror_feedback",
                        "stop_decision": sample_stop_decision,
                    }
                )
                write_task_trace(trace_dir, trace)
                if sample_stop_decision["should_stop"]:
                    stop_after_current_sample = True
                    print(
                        f"  stop=pending_mirror_reliable round={round_index} sample={sample_index} "
                        f"predicted_correctness={sample_stop_decision['predicted_correctness']:.3f} "
                        f"severity={sample_stop_decision['severity_score']:.3f}",
                        flush=True,
                    )
                    break
                if mirror_api_error:
                    api_connection_exit_record = {
                        "stopped": True,
                        "round": round_index,
                        "evolution_iteration": round_index + 1,
                        "sample": sample_index,
                        "stage": "mirror_adversary_feedback",
                        "used_current_round_result": True,
                        "returned_empty_completion": False,
                        "reason": mirror_api_error,
                    }
                    trace["stages"].append(
                        {
                            "stage": "api_connection_limit_exit",
                            "phase": "control",
                            "evolution_iteration": round_index + 1,
                            "round": round_index,
                            "sample": sample_index,
                            "curriculum_stage": sample_curriculum_stage,
                            "api_failure_stage": "mirror_adversary_feedback",
                            "used_current_round_result": True,
                            "returned_empty_completion": False,
                            "final_result": {
                                "passed": result.passed,
                                "result": result.result,
                                "failure_mode": result.failure_mode,
                            },
                            "error": mirror_api_error,
                        }
                    )
                    write_task_trace(trace_dir, trace)
                    print(f"  stop=api_connection_limit stage=mirror_adversary_feedback round={round_index} sample={sample_index}", flush=True)
                    break
            if api_connection_exit_record:
                break
            if round_sample_records:
                memory_update_timing = start_timing()
                memory_before_update = memory_snapshot(memory)
                stop_ready_records = [
                    record
                    for record in round_sample_records
                    if record.get("sample_stop_decision", {}).get("should_stop")
                ]
                selectable_records = stop_ready_records or round_sample_records
                selected_record = max(selectable_records, key=lambda item: item["mirror_selection_score"])
                selected_mirror_feedback = selected_record["mirror_feedback"]
                selected_strategy = selected_record["optimized_strategy"]
                memory.update_strategy_feedback(role, selected_mirror_feedback)
                memory_update_timing_result = finish_timing(memory_update_timing)
                last_result = None
                last_mirror_feedback = selected_mirror_feedback 
                reasoning_strategy = selected_strategy 
                trace["stages"].append(
                    {
                        "stage": "mirror_feedback_best_of_samples_selection",
                        "phase": "micro",
                        "evolution_iteration": round_index + 1,
                        "round": round_index,
                        "selected_sample": selected_record["sample"],
                        "selected_curriculum_stage": selected_record["curriculum_stage"],
                        "selected_candidate_completion": selected_record["candidate"].completion,
                        "selection_basis": "mirror_feedback_only_stop_ready_first"
                        if stop_ready_records
                        else "mirror_feedback_only",
                        "mirror_selection_score": selected_record["mirror_selection_score"],
                        "selected_mirror_feedback": selected_mirror_feedback,
                        "selected_optimized_strategy_for_next_iteration": selected_strategy,
                        "sample_loop_stopped_early": stop_after_current_sample,
                        "memory_before": memory_before_update,
                        "memory_after": memory_snapshot(memory),
                        "timing": memory_update_timing_result,
                    }
                )
                stop_decision = selected_record.get("sample_stop_decision") or mirror_reliability_stop_decision(
                    selected_mirror_feedback,
                    round_index,
                    args,
                )
                trace["stages"].append(
                    {
                        "stage": "mirror_reliability_stop_check",
                        "phase": "control",
                        "evolution_iteration": round_index + 1,
                        "round": round_index,
                        "selected_sample": selected_record["sample"],
                        "selected_curriculum_stage": selected_record["curriculum_stage"],
                        "selection_basis": "mirror_feedback_only_stop_ready_first"
                        if stop_ready_records
                        else "mirror_feedback_only",
                        "sample_loop_stopped_early": stop_after_current_sample,
                        "stop_decision": stop_decision,
                    }
                )
                write_task_trace(trace_dir, trace)
                if stop_decision["should_stop"]:
                    mirror_stop_record = {
                        "stopped": True,
                        "round": round_index,
                        "evolution_iteration": round_index + 1,
                        "selected_sample": selected_record["sample"],
                        "selected_curriculum_stage": selected_record["curriculum_stage"],
                        "stop_decision": stop_decision,
                    }
                    print(
                        f"  stop=mirror_reliable round={round_index} "
                        f"predicted_correctness={stop_decision['predicted_correctness']:.3f} "
                        f"severity={stop_decision['severity_score']:.3f}",
                        flush=True,
                    )
                    break
        results_by_task[problem.task_id] = task_results
        task_timing_result = finish_timing(task_timing)
        stage_time_summary: Dict[str, float] = {}
        for stage in trace["stages"]:
            timing = stage.get("timing", {})
            if "duration_seconds" in timing:
                stage_time_summary[stage["stage"]] = stage_time_summary.get(stage["stage"], 0.0) + timing["duration_seconds"]
        trace["summary"] = {
            "stage": "task_summary",
            "attempts": len(task_results),
            "passed_any_round": any(result.passed for result in task_results),
            "mirror_dynamic_stop": mirror_stop_record or {"stopped": False},
            "api_connection_limit_exit": api_connection_exit_record or {"stopped": False},
            "total_duration_seconds": task_timing_result["duration_seconds"],
            "stage_time_summary": stage_time_summary,
            "final_result": None
            if not task_results
            else {
                "passed": task_results[-1].passed,
                "result": task_results[-1].result,
                "failure_mode": task_results[-1].failure_mode,
            },
        }
        write_task_trace(trace_dir, trace)
        strategy_feedback_added = memory.strategy_feedback[task_strategy_feedback_start:]
        write_hase_sample_result(
            sample_result_path,
            problem,
            index - 1,
            task_sample_rows,
            task_result_rows,
            strategy_feedback_added,
            trace,
        )
        completed_sample_files[problem.task_id] = str(sample_result_path)

    metrics_by_round = compute_metrics_by_round(results_by_round, args.k, max_rounds=args.rounds)
    overall_solved_rate = compute_overall_solved_rate(results_by_task)
    summary = {
        "model": config.model,
        "base_url": config.base_url,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "api_retries": config.api_retries,
        "retry_sleep": config.retry_sleep,
        "connection_failure_limit": config.connection_failure_limit,
        "api_connection_limit_policy": (
            "If one API call has more than connection_failure_limit consecutive connection failures, "
            "the current problem stops evolving. If a candidate already exists in the current round, "
            "that candidate is kept as the final answer; otherwise an empty completion is recorded."
        ),
        "num_tasks": len(problems),
        "rounds": args.rounds,
        "samples_per_round": args.samples_per_round,
        "stop_on_pass": args.stop_on_pass,
        "sample_result_dir": str(per_sample_dir),
        "skipped_existing_tasks": skipped_tasks,
        "completed_sample_files": completed_sample_files,
        "mirror_stop": args.mirror_stop,
        "mirror_stop_thresholds": {
            "mirror_stop_min_rounds": args.mirror_stop_min_rounds,
            "mirror_stop_correctness": args.mirror_stop_correctness,
            "mirror_stop_severity": args.mirror_stop_severity,
            "mirror_stop_confidence": args.mirror_stop_confidence,
            "mirror_stop_blocking_modes": list(args.mirror_stop_blocking_modes),
        },
        "metrics_by_round": metrics_by_round,
        "overall_solved_rate": overall_solved_rate,
        "role_stats": {role.value: stats for role, stats in memory.role_stats.items()},
        "failure_counts": memory.failure_counts,
        "learned_rules": memory.learned_rules,
        "strategy_feedback": memory.strategy_feedback,
    }
    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {
        f"round_{round_index}_{metric}": value
        for round_index, metrics in metrics_by_round.items()
        for metric, value in metrics.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HASE with an external model API on HumanEval.")
    parser.add_argument(
        "--problem-file",
        default="original-datasets/human-eval-master/data/HumanEval.jsonl.gz",
        help="Path to HumanEval.jsonl or HumanEval.jsonl.gz.",
    )
    parser.add_argument(
        "--rounds",
        "--max-rounds",
        dest="rounds",
        type=int,
        default=2,
        help="Maximum evolution rounds per HumanEval problem; mirror stopping may end earlier.",
    )
    parser.add_argument("--samples-per-round", type=int, default=1, help="API samples generated per round.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N problems.")
    parser.add_argument("--eval-timeout", type=float, default=3.0, help="Timeout for executing one candidate.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for candidate execution.")
    parser.add_argument("--seed", type=int, default=None, help="Optional model seed passed to APIs that support a seed field.")
    parser.add_argument("--api-retries", type=int, default=None, help="Retry count for transient API connection/HTTP errors.")
    parser.add_argument("--retry-sleep", type=float, default=None, help="Base sleep seconds between API retries.")
    parser.add_argument(
        "--connection-failure-limit",
        type=int,
        default=None,
        help="Stop evolving the current problem after more than this many consecutive API connection failures; default is HASE_CONNECTION_FAILURE_LIMIT or 20.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/HASE/hase_humaneval",
        help="Output prefix for .samples.jsonl, .results.jsonl, and .summary.json.",
    )
    parser.add_argument(
        "--trace-dir",
        default="results/HASE",
        help="Directory for one detailed JSON trace file per HumanEval sample.",
    )
    parser.add_argument(
        "--sample-result-dir",
        default=None,
        help="Directory for one completed JSON result per HumanEval sample. Existing files are loaded and skipped.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10], help="pass@k values to report.")
    parser.add_argument(
        "--stop-on-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop evolving a problem once a candidate passes.",
    )
    parser.add_argument(
        "--mirror-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dynamically stop evolution when mirror feedback predicts the selected candidate is reliable.",
    )
    parser.add_argument(
        "--mirror-stop-min-rounds",
        type=int,
        default=1,
        help="Minimum completed evolution rounds before mirror reliability stopping can trigger.",
    )
    parser.add_argument(
        "--mirror-stop-correctness",
        type=float,
        default=0.85,
        help="Stop when selected mirror predicted_correctness is at least this value.",
    )
    parser.add_argument(
        "--mirror-stop-severity",
        type=float,
        default=0.20,
        help="Stop when selected mirror severity_score is at most this value.",
    )
    parser.add_argument(
        "--mirror-stop-confidence",
        type=float,
        default=0.0,
        help="Stop when selected mirror confidence is at least this value.",
    )
    parser.add_argument(
        "--mirror-stop-blocking-modes",
        nargs="+",
        default=["syntax_error", "runtime_error", "timeout", "empty_completion", "assertion_error", "edge_case_gap"],
        help="Mirror-predicted failure modes that prevent early stopping.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_hase_humaneval(args)


if __name__ == "__main__":
    main()
