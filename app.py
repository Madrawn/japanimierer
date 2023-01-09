# Thank AK. https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/blob/main/app.py
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPFeatureExtractor
import gradio as gr
import torch
from PIL import Image

model_id = 'aipicasso/cool-japan-diffusion-2-1-0'
prefix = ''
     
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
feature_extractor = CLIPFeatureExtractor.from_pretrained(model_id)

pipe = StableDiffusionPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler)

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler,
  requires_safety_checker=False,
  safety_checker=None,
  feature_extractor=feature_extractor
)

if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe_i2i = pipe_i2i.to("cuda")

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def inference(prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", auto_prefix=False):

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
  prompt = f"{prefix} {prompt}" if auto_prefix else prompt

  try:
    if img is not None:
      return img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None
    else:
      return txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator), None
  except Exception as e:
    return None, error_str(e)

def auto_prompt_correction(prompt_ui,neg_prompt_ui):
    # auto prompt correction
    prompt=str(prompt_ui)
    neg_prompt=str(neg_prompt_ui)
    prompt=prompt.lower()
    neg_prompt=neg_prompt.lower()
    if(prompt=="" and neg_prompt==""):
        prompt="anime, a portrait of a girl, 4k, detailed"
        neg_prompt=" (((deformed))), blurry, ((((bad anatomy)))), bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    human_words=["girl","maid","female","woman","boy","male","man","guy"]
    for word in human_words:
        if( (prompt==f"a {word}" or prompt==word) and neg_prompt==""):
            prompt=f"anime, a portrait of a {word}, 4k, detailed"
            neg_prompt=" (((deformed))), blurry, ((((bad anatomy)))), bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    animal_words=["cat","dog","bird"]
    for word in animal_words:
        if( (prompt==f"a {word}" or prompt==word) and neg_prompt==""):
            prompt=f"anime, a {word}, 4k, detailed"
            neg_prompt=" (((deformed))), blurry, ((((bad anatomy)))), bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    background_words=["mount fuji","mt. fuji","building", "buildings", "tokyo"]
    for word in background_words:
        if( (prompt==f"a {word}" or prompt==word) and neg_prompt==""):
            prompt=f"anime, shinkai makoto, {word}, 4k, 8k, highly detailed"
            neg_prompt=" (((deformed))), photo, people, low quality, ui, error, lowres, jpeg artifacts, 2d, 3d, cg, text"

    return prompt,neg_prompt
    
def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator):
    prompt,neg_prompt=auto_prompt_correction(prompt,neg_prompt)
    
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return result.images[0]

def img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe_i2i(
        prompt,
        negative_prompt = neg_prompt,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        #width = width,
        #height = height,
        generator = generator)
        
    return result.images[0]

css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="main-div">
              <div>
                <h1>Cool Japan Diffusion 2.1.0</h1>
              </div>
              <p>
               Demo for <a href="https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-0">Cool Japan Diffusion 2 1 0</a> Stable Diffusion model.<br>
               {"Add the following tokens to your prompts for the model to work properly: <b>prefix</b>" if prefix else ""}
              </p>
              <p>
              sample prompt: anime, ilya kuvshinov , (evangelion), a portrait of a girl with blue short hair and red eyes, ayanami rei, full color illustration, official art, 4k, detailed
              </p>
              <p>
              sample negative prompt: low quality, bad face, ((((bad anatomy)))), ((bad hand)), lowres, jpeg artifacts, 2d, 3d, cg, text
              </p>
              <p>
              <a href="https://alfredplpl.hatenablog.com/entry/2022/12/30/102636">日本語の取扱説明書</a>.
              </p>
              Running on {"<b>GPU 🔥</b>" if torch.cuda.is_available() else f"<b>CPU 🥶</b>. For faster inference it is recommended to <b>upgrade to GPU in <a href='https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/settings'>Settings</a></b>"} after duplicating the space<br><br>
              <a style="display:inline-block" href="https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder=f"{prefix} [your prompt]").style(container=False)
                generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))

              image_out = gr.Image(height=512)
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            with gr.Group():
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
              auto_prefix = gr.Checkbox(label="Prefix styling tokens automatically ()", value=prefix, visible=prefix)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                steps = gr.Slider(label="Steps", value=20, minimum=2, maximum=75, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

              seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

          with gr.Tab("Image to image"):
              with gr.Group():
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    auto_prefix.change(lambda x: gr.update(placeholder=f"{prefix} [your prompt]" if x else "[Your prompt]"), inputs=auto_prefix, outputs=prompt, queue=False)

    inputs = [prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, auto_prefix]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>This space was created using <a href="https://huggingface.co/spaces/anzorq/sd-space-creator">SD Space Creator</a>.</p>
    </div>
    """)

demo.queue(concurrency_count=1)
demo.launch()