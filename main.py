from gen_table import generate_table_files
from bench import run_benchmark

prompt = "What geographical area does this resemble? Answer with only the name of the place. Answer quick, don't think too much!"

models = [
    "openrouter/openai/gpt-5.2"
    "openrouter/google/gemini-3-pro-preview",
    "openrouter/anthropic/claude-opus-4.5",
    "openrouter/x-ai/grok-4-fast",
    "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
# Could also test
    # "openrouter/z-ai/glm-4.6v",
    # "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    # "openrouter/google/gemma-3-27b-it",
    # "openrouter/google/gemini-2.5-flash",
    # "openrouter/qwen/qwen3-vl-8b-instruct",
]

def main():
    if run_benchmark(
        prompt=prompt,
        models=models
    ):
        print("Benchmark completed successfully.")
        generate_table_files(
            do_accuracy=False,
            do_models=True,
            do_answers=False
        )
    
if __name__ == "__main__":
    main()
