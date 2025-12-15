from dataclasses import dataclass
from html import escape
import re
import os
from pathlib import Path
from typing import Sequence

from inspect_ai.log import EvalLog, read_eval_log


@dataclass(slots=True)
class ModelSummary:
    name: str
    num_correct: int
    total_samples: int

    @property
    def accuracy(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.num_correct / self.total_samples


@dataclass(slots=True)
class SampleResult:
    filename: str
    completion: str
    is_correct: bool
    image_path: Path
    targets: tuple[str, ...]


def get_eval_logs(path: Path) -> list[EvalLog]:
    if not path.is_dir():
        raise FileNotFoundError(f"Log directory not found: {path}")

    eval_files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix == ".eval")
    if not eval_files:
        raise FileNotFoundError(f"No eval log files found in {path}")

    return [read_eval_log(f) for f in eval_files]


def _sample_is_correct(sample: object) -> bool:
    score = getattr(sample, "score", None)
    if score is None:
        return False

    value = getattr(score, "value", None)
    if isinstance(value, str):
        return value.lower() == "c"
    if isinstance(value, bool):
        return value

    metrics = getattr(score, "metrics", None)
    if isinstance(metrics, dict):
        accuracy_metric = metrics.get("accuracy")
        metric_value = getattr(accuracy_metric, "value", None)
        if isinstance(metric_value, (int, float)):
            return metric_value > 0
        if metric_value is not None:
            return bool(metric_value)

    answer = getattr(score, "answer", None)
    if isinstance(answer, str):
        normalized_answer = answer.strip().lower()
        targets = [str(t).strip().lower() for t in getattr(sample, "target", [])]
        return normalized_answer in targets

    raise ValueError("Unable to determine correctness for sample")


def _sample_filename(sample: object) -> str:
    # Prefer filename stored on the input message metadata (matches bench dataset)
    inputs = getattr(sample, "input", None) or []
    if inputs:
        metadata = getattr(inputs[0], "metadata", None) or {}
        filename = metadata.get("filename")
        if filename:
            return filename

    sample_metadata = getattr(sample, "metadata", None) or {}
    filename = sample_metadata.get("filename")
    if filename:
        return filename

    sample_id = getattr(sample, "id", None)
    if sample_id:
        return str(sample_id)

    raise ValueError("Sample is missing a filename identifier")


def _sample_completion(sample: object) -> str:
    output = getattr(sample, "output", None)
    completion = getattr(output, "completion", None)
    if completion:
        return completion

    score = getattr(sample, "score", None)
    explanation = getattr(score, "explanation", None)
    if explanation:
        return explanation

    return ""


def _collect_sample_results(log: EvalLog, images_dir: Path) -> tuple[ModelSummary, list[SampleResult]]:
    model_name = str(getattr(log.eval, "model", "unknown"))
    samples = list(getattr(log, "samples", []))

    sample_results: list[SampleResult] = []
    for sample in samples:
        filename = _sample_filename(sample)
        image_path = images_dir / filename
        completion = _sample_completion(sample)
        is_correct = _sample_is_correct(sample)
        targets_raw = list(getattr(sample, "target", []))
        targets = tuple(str(target) for target in targets_raw)

        sample_results.append(
            SampleResult(
                filename=filename,
                completion=completion,
                is_correct=is_correct,
                image_path=image_path,
                targets=targets,
            )
        )

    num_correct = sum(1 for sample in sample_results if sample.is_correct)
    total_samples = len(sample_results)

    return ModelSummary(model_name, num_correct, total_samples), sample_results


def _escape_html_text(value: str) -> str:
    # Preserve intentional line breaks inside table cells.
    return escape(value or "", quote=False).replace("\n", "<br>").strip()


def _escape_html_attr(value: str) -> str:
    return escape(value or "", quote=True)


def _format_targets(targets: Sequence[str]) -> str:
    targets_text = _escape_html_text(", ".join(targets))
    return (
        "<details>"
        "<summary>Show answer</summary>"
        f"<div><strong>{targets_text}</strong></div>"
        "</details>"
    )


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    column_widths: Sequence[str] | None = None,
) -> str:
    if column_widths and len(column_widths) != len(headers):
        raise ValueError("Column widths must match headers length.")
    header_cells = "".join(f"<th>{_escape_html_text(str(header))}</th>" for header in headers)
    colgroup = ""
    if column_widths:
        col_tags = "".join(f'    <col width="{_escape_html_attr(width)}">\n' for width in column_widths)
        colgroup = "<colgroup>\n" + col_tags + "</colgroup>\n"
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"    <tr>{cells}</tr>")
    body = "\n".join(body_rows)
    if body:
        body += "\n"
    return (
        f"<table width=\"100%\">\n"
        f"{colgroup}"
        "  <thead>\n"
        f"    <tr>{header_cells}</tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        f"{body}"
        "  </tbody>\n"
        "</table>"
    )


def _slugify(name: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "-" for char in name)
    condensed = "-".join(filter(None, sanitized.split("-")))
    return condensed or "model"


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def _build_scoreboard_content(summaries: Sequence[ModelSummary]) -> str:
    sorted_summaries = sorted(
        summaries,
        key=lambda summary: (-summary.accuracy, -summary.num_correct, summary.name),
    )

    rows = [
        (
            _escape_html_text(summary.name.removeprefix("openrouter/")),
            _escape_html_text(f"{summary.num_correct}/{summary.total_samples}"),
        )
        for summary in sorted_summaries
    ]

    table = _render_table(("Model", "Accuracy"), rows, column_widths=("70%", "30%"))
    return "# Model Accuracy\n\n" + table + "\n"


def _build_model_table_content(
    model_name: str,
    samples: Sequence[SampleResult],
    output_dir: Path,
) -> str:
    rows: list[tuple[str, str, str]] = []
    for sample in samples:
        image_link = _relpath(sample.image_path, start=output_dir)
        filename_cell = (
            f'<a href="{_escape_html_attr(image_link)}">{_escape_html_text(sample.filename)}</a>'
        )
        answer_cell = _escape_html_text(sample.completion)
        correctness_cell = "✅" if sample.is_correct else "❌"
        rows.append((filename_cell, answer_cell, correctness_cell))

    table = _render_table(
        ("Filename", "Answer", "Correct?"),
        rows,
        column_widths=("35%", "45%", "20%"),
    )
    return f"# {model_name}\n\n{table}\n"


def _build_study_table_content(samples: Sequence[SampleResult], output_dir: Path) -> str:
    rows: list[tuple[str, str]] = []
    for sample in samples:
        image_src = _escape_html_attr(_relpath(sample.image_path, start=output_dir))
        image_alt = _escape_html_attr(sample.filename)
        image_markdown = f'<img src="{image_src}" alt="{image_alt}" width="500">'
        targets_cell = _format_targets(sample.targets)
        rows.append((image_markdown, targets_cell))

    table = _render_table(
        ("Image", "Targets"),
        rows,
        column_widths=("45%", "55%"),
    )
    return "# Answers\n\n" + table + "\n"


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        relative = path
    print(f"Wrote {relative}")


def generate_table_files(
    log_dir: Path = Path("logs"),
    images_dir: Path = Path("images"),
    output_dir: Path = Path("tables"),
) -> None:
    eval_logs = get_eval_logs(log_dir)

    summaries: dict[str, ModelSummary] = {}
    model_samples: dict[str, list[SampleResult]] = {}
    study_samples: dict[str, SampleResult] = {}

    for log in eval_logs:
        summary, samples = _collect_sample_results(log, images_dir)
        existing_summary = summaries.get(summary.name)
        if existing_summary:
            summaries[summary.name] = ModelSummary(
                name=summary.name,
                num_correct=existing_summary.num_correct + summary.num_correct,
                total_samples=existing_summary.total_samples + summary.total_samples,
            )
        else:
            summaries[summary.name] = summary

        model_samples.setdefault(summary.name, []).extend(samples)
        for sample in samples:
            study_samples.setdefault(sample.filename, sample)

    output_dir.mkdir(parents=True, exist_ok=True)

    scoreboard_path = output_dir / "model-accuracy.md"
    _write_markdown(scoreboard_path, _build_scoreboard_content(list(summaries.values())))

    for model_name, samples in model_samples.items():
        model_path = output_dir / f"{_slugify(model_name)}.md"
        ordered_samples = sorted(samples, key=lambda sample: _filename_sort_key(sample.filename))
        _write_markdown(model_path, _build_model_table_content(model_name, ordered_samples, output_dir))

    study_path = output_dir / "answers.md"
    ordered_study_samples = sorted(study_samples.values(), key=lambda sample: sample.filename.lower())
    _write_markdown(
        study_path,
        _build_study_table_content(ordered_study_samples, output_dir),
    )
def _filename_sort_key(filename: str) -> tuple:
    parts = re.findall(r"\d+|\D+", filename.lower())
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)
