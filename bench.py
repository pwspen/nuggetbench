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

# Answers are stored lowercased and without spaces
# But we also want to keep the raw completion
@scorer(metrics=[accuracy(), stderr()])
def custom_scorer() -> Callable:
    async def score(state: TaskState, target: Target) -> Score:
        raw: str = state.output.completion
        answer: str = _normalize(raw)
        targets: list[str] = [_normalize(t) for t in target]

        return Score(
            value=CORRECT if answer in targets else INCORRECT,
            answer=answer,
            explanation=raw,
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
    """
    Compiles all samples (from images folder) into a MemoryDataset
    Extracts targets from image filenames
    Keeps image path as metadata
    """
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir} (extensions: {sorted(IMAGE_EXTS)})")

    samples: list[Sample] = []
    for image_path in images:
        # Labels are stored in filenames to ensure image-label mismatch can never happen
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
            )
        )

    return MemoryDataset(samples)

def run_benchmark(
        prompt: str,
        models: list[str],
        system: str = "You are a helpful assistant. Do as the user asks.",
        images_dir: str = "images/",
        temperature: float = 0.7,
        max_tokens: int = 30000,
) -> bool:
    """
    Runs benchmark on given models with images from images_dir (which must follow naming convention) and prompt
    """

    @task
    def create_task(image_dir: str, prompt: str) -> Task:
        return Task(
            dataset=dataset_from_image_folder(Path(image_dir), prompt),
            solver=[
                system_message(system),
                generate(temperature=temperature, max_tokens=max_tokens),
            ],
            scorer=custom_scorer(),
        )

    result = eval(
        create_task(image_dir=images_dir, prompt=prompt),
        model=models,
    )[0]
    
    return result.status == "success"