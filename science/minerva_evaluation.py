from __future__ import annotations

import json
import math
import re
import ast
import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CATEGORY_ORDER = ("chemistry", "mathematics", "physics", "other")

LATEX_COMMAND_NAMES = (
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|"
    r"omicron|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    r"frac|dfrac|tfrac|sqrt|left|right|text|mathrm|textrm|mathbf|mathit|operatorname|"
    r"boxed|fbox|begin|end|overline|underline|bar|vec|hat|dot|ddot|ell|hbar|"
    r"cdot|times|odot|otimes|pm|mp|le|leq|ge|geq|neq|approx|sim|equiv|propto|infty|"
    r"sum|prod|int|iint|iiint|lim|partial|nabla|"
    r"sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp|min|max|det|Re|Im"
)
DUPLICATE_LATEX_ESCAPE_RE = re.compile(rf"\\{{2,}}(?=(?:{LATEX_COMMAND_NAMES})(?:\b|(?=\s*\{{)))")
DUPLICATE_LATEX_SYMBOL_ESCAPE_RE = re.compile(r"\\{2,}(?=[\[\](),;!:])")

CATEGORY_TYPES = {
    "chemistry": {
        "Introduction to Solid State Chemistry (3.091 Fall 2010)",
        "Physical Chemistry (5.61 Fall 2017)",
    },
    "mathematics": {
        "Differential Equations (18.03 Spring 2010)",
        "Dynamics and Control (2.003 Spring 2005)",
        "Information and Entropy (6.050J Spring 2008)",
    },
    "physics": {
        "Introduction to Astronomy (8.282J Spring 2006)",
        "Relativity (8.033 Fall 2006)",
    },
    "other": {
        "Ecology I (1.018J Fall 2009)",
        "Principles of Microeconomics (14.01 Fall 2011)",
    },
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def category_for_type(type_name: str) -> str:
    normalized = str(type_name).strip()
    for category in CATEGORY_ORDER:
        if normalized in CATEGORY_TYPES[category]:
            return category
    return "other"


def _balanced_brace_content(text: str, open_index: int) -> Optional[Tuple[str, int]]:
    if open_index < 0 or open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 0
    content_start = open_index + 1
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
    return None


def extract_boxed_answer(text: str) -> str:
    candidates: List[str] = []
    for marker in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            marker_index = text.find(marker, start)
            if marker_index < 0:
                break
            brace_index = text.find("{", marker_index + len(marker))
            if brace_index < 0:
                break
            parsed = _balanced_brace_content(text, brace_index)
            if parsed is None:
                start = brace_index + 1
                continue
            content, end_index = parsed
            candidates.append(content.strip())
            start = end_index
    return candidates[-1] if candidates else ""


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    patterns = (
        r"(?is)final\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        r"(?is)answer\s*(?:is|:|=)\s*([^\n]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).strip().rstrip(".。")
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1].rstrip(".。") if non_empty_lines else ""


def _replace_latex_fraction(text: str) -> str:
    pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    current = text
    while current != previous:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _replace_latex_sqrt(text: str) -> str:
    pattern = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    previous = None
    current = text
    while current != previous:
        previous = current
        current = pattern.sub(r"sqrt(\1)", current)
    return current


def _remove_latex_text_commands(text: str) -> str:
    pattern = re.compile(r"\\(?:mathrm|textrm|text|operatorname)\s*\{([^{}]*)\}")
    previous = None
    current = text
    while current != previous:
        previous = current
        current = pattern.sub(r"\1", current)
    return current


def strip_assignment(text: str) -> str:
    if text.count("=") != 1:
        return text
    left, right = text.split("=", 1)
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*(?:\([^)]*\))?", left.strip()):
        return right.strip()
    return text


def normalize_answer(answer: str) -> str:
    text = str(answer or "").strip()
    text = DUPLICATE_LATEX_ESCAPE_RE.sub(lambda _: "\\", text)
    text = DUPLICATE_LATEX_SYMBOL_ESCAPE_RE.sub(lambda _: "\\", text)
    text = text.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\pi", "pi").replace("π", "pi")
    for latex_name, plain_name in (
        (r"\arcsin", "asin"),
        (r"\arccos", "acos"),
        (r"\arctan", "atan"),
        (r"\sin", "sin"),
        (r"\cos", "cos"),
        (r"\tan", "tan"),
        (r"\ln", "log"),
        (r"\log", "log"),
        (r"\exp", "exp"),
    ):
        text = text.replace(latex_name, plain_name)
    text = text.replace("np.", "")
    text = re.sub(r"\barcsin\b", "asin", text)
    text = re.sub(r"\barccos\b", "acos", text)
    text = re.sub(r"\barctan\b", "atan", text)
    text = re.sub(r"\^?\\circ", "", text)
    text = text.replace("$", "").replace("\n", " ")
    text = _replace_latex_fraction(text)
    text = _replace_latex_sqrt(text)
    text = _remove_latex_text_commands(text)
    text = strip_assignment(text)
    text = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", text)
    text = re.sub(r"10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?", r"10^(\1)", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.strip(".,;:。；：")
    text = re.sub(r"(?i)(degrees?|radians?|ergs?(?:/s|s\^-1)?|cm|mm|km|meters?|metres?|seconds?|secs?|kelvin|k)$", "", text)
    text = re.sub(r"(?i)(\\circ|°)$", "", text)
    return text.strip()


def _safe_numeric_expression(text: str) -> Optional[float]:
    normalized = normalize_answer(text).replace("^", "**")
    if not normalized or len(normalized) > 300:
        return None
    allowed_functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "ln": math.log,
        "exp": math.exp,
        "abs": abs,
    }
    allowed_constants = {"pi": math.pi, "e": math.e}

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in allowed_constants:
            return float(allowed_constants[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and abs(right) <= 1000:
                return left**right
            raise ValueError("unsupported binary operation")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in allowed_functions
            and len(node.args) == 1
            and not node.keywords
        ):
            return float(allowed_functions[node.func.id](evaluate(node.args[0])))
        raise ValueError("unsupported expression")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.parse(normalized, mode="eval")
        value = float(evaluate(parsed))
        return value if math.isfinite(value) else None
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _numeric_prefix(text: str) -> Optional[float]:
    normalized = normalize_answer(text)
    percent = normalized.endswith("%")
    if percent:
        normalized = normalized[:-1]
    scientific = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\*?10\^\(?([+-]?\d+)\)?)",
        normalized,
    )
    if scientific:
        value = float(scientific.group(1)) * (10.0 ** int(scientific.group(2)))
        return value / 100.0 if percent else value
    plain = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", normalized)
    if plain:
        value = float(normalized)
        return value / 100.0 if percent else value
    fraction = re.fullmatch(
        r"\(?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)?/\(?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)?",
        normalized,
    )
    if fraction and float(fraction.group(2)) != 0.0:
        value = float(fraction.group(1)) / float(fraction.group(2))
        return value / 100.0 if percent else value
    expression_value = _safe_numeric_expression(normalized)
    if expression_value is not None:
        return expression_value / 100.0 if percent else expression_value
    return None


def _reference_precision_tolerance(reference: str, reference_number: float) -> Optional[float]:
    normalized = normalize_answer(reference)
    percent = normalized.endswith("%")
    if percent:
        normalized = normalized[:-1]
    scale = 0.01 if percent else 1.0

    scientific = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\*?10\^\(?([+-]?\d+)\)?)",
        normalized,
    )
    if scientific:
        coefficient = scientific.group(1)
        exponent = int(scientific.group(2))
        if "." not in coefficient:
            return None
        decimal_places = len(coefficient.split(".", 1)[1])
        return 0.5 * (10.0 ** (exponent - decimal_places)) * scale

    plain = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", normalized)
    if plain and "." in normalized and "e" not in normalized.lower():
        decimal_places = len(normalized.split(".", 1)[1])
        return 0.5 * (10.0 ** (-decimal_places)) * scale

    return None


def _sympy_expression(text: str):
    try:
        import sympy
    except ImportError:
        return None
    raw = str(text or "").strip().strip("$")
    try:
        from sympy.parsing.latex import parse_latex

        parsed_latex = parse_latex(raw)
        if parsed_latex is not None:
            return parsed_latex
    except Exception:
        pass
    normalized = normalize_answer(text)
    if not normalized:
        return None
    normalized = normalized.replace("^", "**")
    normalized = re.sub(r"(?<=\d)pi", "*pi", normalized)
    normalized = re.sub(r"(?<=\d)\(", "*(", normalized)
    normalized = re.sub(r"\)(?=\d|[a-zA-Z])", ")*", normalized)
    try:
        return sympy.sympify(normalized, locals={"pi": sympy.pi, "sqrt": sympy.sqrt})
    except Exception:
        return None


def evaluator_capabilities() -> Dict[str, Any]:
    sympy_available = importlib.util.find_spec("sympy") is not None
    antlr_available = importlib.util.find_spec("antlr4") is not None
    return {
        "normalized_exact_match": True,
        "numeric_tolerance": True,
        "reference_precision_numeric_tolerance": True,
        "safe_numeric_expression": True,
        "sympy_available": sympy_available,
        "antlr_latex_parser_available": bool(sympy_available and antlr_available),
        "symbolic_equivalence": "sympy_latex_and_simplify" if sympy_available else "normalized_exact_fallback",
    }


def answers_equivalent(
    prediction: str,
    reference: str,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    predicted_normalized = normalize_answer(prediction)
    reference_normalized = normalize_answer(reference)
    base = {
        "prediction": prediction,
        "reference": reference,
        "normalized_prediction": predicted_normalized,
        "normalized_reference": reference_normalized,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }
    if not predicted_normalized or not reference_normalized:
        return {**base, "correct": False, "match_method": "empty_answer"}
    if predicted_normalized.lower() == reference_normalized.lower():
        return {**base, "correct": True, "match_method": "normalized_exact_match"}

    predicted_number = _numeric_prefix(prediction)
    reference_number = _numeric_prefix(reference)
    if predicted_number is not None and reference_number is not None:
        strict_tolerance = max(
            absolute_tolerance,
            relative_tolerance * max(abs(reference_number), abs(predicted_number), 1.0),
        )
        reference_precision_tolerance = _reference_precision_tolerance(reference, reference_number)
        tolerance = strict_tolerance
        match_method = "numeric_tolerance"
        if reference_precision_tolerance is not None and reference_precision_tolerance > tolerance:
            tolerance = reference_precision_tolerance
            match_method = "reference_precision_numeric_tolerance"
        correct = math.isfinite(predicted_number) and math.isfinite(reference_number)
        correct = correct and abs(predicted_number - reference_number) <= tolerance
        return {
            **base,
            "correct": bool(correct),
            "match_method": match_method,
            "predicted_number": predicted_number,
            "reference_number": reference_number,
            "strict_numeric_tolerance": strict_tolerance,
            "reference_precision_tolerance": reference_precision_tolerance,
            "numeric_tolerance": tolerance,
        }

    predicted_expression = _sympy_expression(prediction)
    reference_expression = _sympy_expression(reference)
    if predicted_expression is not None and reference_expression is not None:
        try:
            import sympy

            if isinstance(predicted_expression, sympy.Equality):
                predicted_expression = predicted_expression.lhs - predicted_expression.rhs
            if isinstance(reference_expression, sympy.Equality):
                reference_expression = reference_expression.lhs - reference_expression.rhs
            correct = bool(sympy.simplify(predicted_expression - reference_expression) == 0)
            return {**base, "correct": correct, "match_method": "sympy_equivalence"}
        except Exception:
            pass
    return {**base, "correct": False, "match_method": "not_equivalent"}


def evaluate_solution(
    model_solution: str,
    reference_solution: str,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    prediction = extract_final_answer(model_solution)
    reference = extract_final_answer(reference_solution)
    comparison = answers_equivalent(
        prediction,
        reference,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return {
        "metric": "accuracy",
        "scoring_rule": "final_answer_mathematical_equivalence",
        "predicted_answer": prediction,
        "reference_answer": reference,
        **comparison,
    }


def _empty_metric_block() -> Dict[str, Any]:
    return {"correct": 0, "total": 0, "accuracy": 0.0, "accuracy_percent": 0.0}


def metric_block(outcomes: Iterable[Tuple[str, bool]]) -> Dict[str, Any]:
    categories = {category: _empty_metric_block() for category in CATEGORY_ORDER}
    overall = _empty_metric_block()
    for category, correct in outcomes:
        selected_category = category if category in categories else "other"
        categories[selected_category]["total"] += 1
        overall["total"] += 1
        if correct:
            categories[selected_category]["correct"] += 1
            overall["correct"] += 1
    for block in [overall, *categories.values()]:
        block["accuracy"] = block["correct"] / block["total"] if block["total"] else 0.0
        block["accuracy_percent"] = block["accuracy"] * 100.0
    return {
        "official_metric": "accuracy",
        "overall": overall,
        "average_accuracy": overall["accuracy"],
        "categories": categories,
    }


def _round_record_for(result: Dict[str, Any], round_index: int) -> Optional[Dict[str, Any]]:
    rounds = [item for item in result.get("rounds", []) if isinstance(item, dict)]
    if not rounds:
        return None
    exact = next((item for item in rounds if int(item.get("round", 0)) == round_index), None)
    if exact is not None:
        return exact
    prior = [item for item in rounds if int(item.get("round", 0)) <= round_index]
    if prior:
        return max(prior, key=lambda item: int(item.get("round", 0)))
    return min(rounds, key=lambda item: int(item.get("round", 0)))


def aggregate_sample_results(
    results: Sequence[Dict[str, Any]],
    max_rounds: int,
) -> Dict[str, Any]:
    final_outcomes: List[Tuple[str, bool]] = []
    completed = 0
    partial = 0
    for result in results:
        rounds = result.get("rounds", [])
        if not rounds:
            continue
        if not result.get("completed"):
            partial += 1
            continue
        evaluation = result.get("final_official_evaluation") or rounds[-1].get("official_evaluation", {})
        final_outcomes.append((str(result.get("category", "other")), bool(evaluation.get("correct"))))
        completed += 1

    by_round: List[Dict[str, Any]] = []
    for round_index in range(1, max_rounds + 1):
        round_outcomes: List[Tuple[str, bool]] = []
        carried_forward = 0
        for result in results:
            record = _round_record_for(result, round_index)
            if record is None:
                continue
            if int(record.get("round", 0)) < round_index:
                carried_forward += 1
            evaluation = record.get("official_evaluation", {})
            round_outcomes.append((str(result.get("category", "other")), bool(evaluation.get("correct"))))
        metrics = metric_block(round_outcomes)
        metrics.update(
            {
                "round": round_index,
                "available_samples": len(round_outcomes),
                "carried_forward_after_early_stop": carried_forward,
            }
        )
        by_round.append(metrics)

    return {
        "dataset": "Minerva Math",
        "split": "test",
        "official_metric": "accuracy",
        "scoring_rule": "final_answer_mathematical_equivalence",
        "available_sample_results": len(final_outcomes),
        "completed_sample_results": completed,
        "partial_sample_results": partial,
        "final_metrics": metric_block(final_outcomes),
        "metrics_by_round": by_round,
    }


def format_metric_line(label: str, metrics: Dict[str, Any]) -> str:
    overall = metrics.get("overall", {})
    categories = metrics.get("categories", {})
    parts = [
        f"{label}",
        f"overall={overall.get('correct', 0)}/{overall.get('total', 0)} ({overall.get('accuracy_percent', 0.0):.2f}%)",
    ]
    for category in CATEGORY_ORDER:
        block = categories.get(category, {})
        parts.append(
            f"{category}={block.get('correct', 0)}/{block.get('total', 0)} "
            f"({block.get('accuracy_percent', 0.0):.2f}%)"
        )
    return " ".join(parts)
