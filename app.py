import os
import gradio as gr

API_KEY=os.environ.get('HUGGING_FACE_HUB_TOKEN', None)

article = """---
This space was created using [SD Space Creator](https://huggingface.co/spaces/anzorq/sd-space-creator)."""

gr.Interface.load(
    name="models/aipicasso/cool-japan-diffusion-2-1-0",
    title="""Cool Japan Diffusion 2.1.0""",
    description="""Demo for <a href="https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-0">Cool Japan Diffusion 2 1 0</a> Stable Diffusion model.""",
    article=article,
    api_key=API_KEY,
    ).queue(concurrency_count=20).launch()
