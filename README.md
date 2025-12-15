How well can LLMs recognize what geographical areas that chicken nuggets resemble?

<table width="100%">
<colgroup>
    <col width="70%">
    <col width="30%">
</colgroup>
  <thead>
    <tr><th>Model</th><th>Accuracy</th></tr>
  </thead>
  <tbody>
    <tr><td>google/gemini-3-pro-preview</td><td>9/18</td></tr>
    <tr><td>anthropic/claude-opus-4.5</td><td>7/18</td></tr>
    <tr><td>qwen/qwen3-vl-235b-a22b-instruct</td><td>5/18</td></tr>
    <tr><td>x-ai/grok-4-fast</td><td>4/18</td></tr>
    <tr><td>openai/gpt-5.2</td><td>2/18</td></tr>
  </tbody>
</table>

![](./images/12_argentina.png)

What does this look like to you?

On one hand, it's a chicken nugget. On the other, it.. looks strangely familiar?

To GPT-5, it's Great Britain. To Gemini 3 and Qwen 3, it's Italy. Grok 4 thinks it's Taiwan. Opus gets it right: Argentina.

This benchmark uses all the images I can find on the internet that show chicken nuggets that are clearly shaped like prominent geographical regions (US states, countries, and continents).

Today, we're in the benchmaxxing, [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) era of AI progress. If it can be verified, it will be trained on. This causes models to be better at things that are commonly used as measures of their intelligence, but it's unclear to what extent the capability gain from training on narrow tasks applies outside of that domain (like it would for humans). For example, models are fantastic at reading text, but horrible at basic visual tasks.

This benchmark tests for something that is pointless and stupid to train for, while also requiring visual acuity and world knowledge. The hope is that this gives a better check of model ability than more sensible or common measures.

See [/tables](./tables) for per-model results.

See [/tables/answers.md](./tables/answers.md) for the dataset and to try it for yourself.

To run the benchmark for yourself, clone this repo. You must have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, and an `OPENROUTER_API_KEY` set as an environment variable. Then, do `uv run main.py`.