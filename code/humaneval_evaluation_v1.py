from __future__ import annotations

import ast
import gzip
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


EVALUATION_VERSION = "v1"


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    result: str
    failure_mode: str
    stderr_tail: str = ""
    execution_time: float = 0.0


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
            import time

            started = time.perf_counter()
            result = subprocess.run(
                [python_executable, str(candidate_file)],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            execution_time = time.perf_counter() - started
        except subprocess.TimeoutExpired:
            return EvalResult(problem.task_id, False, "timed out", "timeout", execution_time=timeout)

    if result.returncode == 0:
        return EvalResult(problem.task_id, True, "passed", "passed", execution_time=execution_time)

    stderr = result.stderr[-1200:] if result.stderr else ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        mode = "syntax_error"
    elif "AssertionError" in stderr:
        mode = "assertion_error"
    else:
        mode = "runtime_error"
    return EvalResult(problem.task_id, False, mode, mode, stderr, execution_time=execution_time)


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
