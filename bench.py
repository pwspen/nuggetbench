from pathlib import Path
from typing import Final, Callable
import re


from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import generate, system_message, TaskState

IMAGE_EXTS: Final[set[str]] = {".png", ".jpg", ".jpeg", ".webp"}

def _normalize(s: str) -> str:
    # Remove whitespace and lowercase
    return re.sub(r"\s+", "", s).lower()

@scorer(metrics=[accuracy(), stderr()])
def custom_scorer() -> Callable:
    async def score(state: TaskState, target: Target) -> Score:
        raw: str = state.output.completion
        answer: str = _normalize(raw)
        targets: list[str] = [_normalize(t) for t in target]

        return Score(
            value=CORRECT if answer in targets else INCORRECT,
            answer=answer,          # store normalized answer
            explanation=raw,        # keep raw completion for debugging
        )

    return score

def label_from_filename(image_path: Path) -> list[str]:
    """
    Filename convention: 0_label1_label2_labelN.ext
    Example: 7_america_usa_unitedstates.jpg
    Returns list of labels
    """
    parts = image_path.stem.split("_")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Bad filename (need at least two '_' delimited parts): {image_path.name}")
    return parts[1:]

def dataset_from_image_folder(image_dir: Path, prompt: str) -> MemoryDataset:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir} (extensions: {sorted(IMAGE_EXTS)})")

    samples: list[Sample] = []
    for image_path in images:
        target = label_from_filename(image_path)
        input_messages: list[ChatMessage] = [
            ChatMessageUser(
                content=[
                    ContentImage(image=str(image_path.resolve())),
                    ContentText(text=prompt),
                ],
                metadata={"filename": image_path.name},
            )
        ]
        samples.append(
            Sample(
                id=image_path.name,
                input=input_messages,
                target=target,
                metadata={"filename": image_path.name},
            )
        )

    return MemoryDataset(samples)


SYSTEM: Final[str] = ""


@task
def create_task(image_dir: str, prompt: str) -> Task:
    return Task(
        dataset=dataset_from_image_folder(Path(image_dir), prompt),
        solver=[
            system_message(SYSTEM),
            generate(temperature=0.0, max_tokens=5000),
        ],
        scorer=custom_scorer(),
    )


def run_benchmark() -> bool:
    # Must have OPENROUTER_API_KEY set in environment!
    MODELS: Final[list[str]] = [
        "openrouter/openai/gpt-5.2",
        "openrouter/google/gemini-3-pro-preview",
        "openrouter/anthropic/claude-opus-4.5",
        "openrouter/x-ai/grok-4-fast",
        "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
    ]

    result = eval(
        create_task(image_dir="images/", prompt="What geographical area does this resemble? Answer with only the name of the place."),
        model=MODELS,
    )[0]
    
    return result.status == "success"