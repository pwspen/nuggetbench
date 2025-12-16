Common backend for image + text in, text-out benchmarks. Includes benchmarking script using InspectAI and markdown table generation for result visualization.

Example use if properly named images are already in "images" folder:
```python
from core.gen_tables import generate_table_files
from core.bench import run_benchmark

prompt = "What geographical area does this resemble? Answer with only the name of the place. Answer quick, don't think too much!"

models = [
    "openrouter/openai/gpt-5.2"
    "openrouter/google/gemini-3-pro-preview",
]

if run_benchmark( # Returns True on success
    prompt=prompt,
    models=models
): # Other args: system, images_dir, log_dir, temperature, max_tokens

    print("Benchmark completed successfully.")
    generate_table_files(
        do_accuracy=True,
        do_models=True,
        do_answers=True
    )
    # Requires paths for logs, images, and output folders - pass as args log_dir, images_dir, output_dir, defaults are "logs/", "images/", "tables/" respectively
```