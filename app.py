# Thank AK. https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/blob/main/app.py
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPFeatureExtractor
import gradio as gr
import torch
from PIL import Image

model_id = 'aipicasso/cool-japan-diffusion-2-1-1-beta'

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


def inference(prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", disable_auto_prompt_correction=False):

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
    
  try:
    if img is not None:
      return img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator, disable_auto_prompt_correction), None
    else:
      return txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator, disable_auto_prompt_correction), None
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
        neg_prompt=f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    splited_prompt=prompt.replace(","," ").replace("_"," ").split(" ")
    splited_prompt=["a person" if p=="solo" else p for p in splited_prompt]
    splited_prompt=["girl" if p=="1girl" else p for p in splited_prompt]
    splited_prompt=["boy" if p=="1boy" else p for p in splited_prompt]
    human_words=["girl","maid","female","woman","boy","male","man","guy"]
    for word in human_words:
        if( word in splited_prompt):
            prompt=f"anime, {prompt}, 4k, detailed"
            neg_prompt=f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    animal_words=["cat","dog","bird"]
    for word in animal_words:
        if( word in splited_prompt):
            prompt=f"anime, a {word}, 4k, detailed"
            neg_prompt=f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, ((mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    background_words=["mount fuji","mt. fuji","building", "buildings", "tokyo", "kyoto", "shibuya", "shinjuku"]
    for word in background_words:
        if( word in splited_prompt):
            prompt=f"anime, shinkai makoto, {word}, 4k, 8k, highly detailed"
            neg_prompt=f"(((deformed))), {neg_prompt}, photo, people, low quality, ui, error, lowres, jpeg artifacts, 2d, 3d, cg, text"

    return prompt,neg_prompt
    
def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator,disable_auto_prompt_correction):
    if(not disable_auto_prompt_correction):
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

def img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator,disable_auto_prompt_correction):
    if(not disable_auto_prompt_correction):
        prompt,neg_prompt=auto_prompt_correction(prompt,neg_prompt)
        
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
                <h1>Cool Japan Diffusion 2.1.1 Beta</h1>
              </div>
              <p>
               Demo for <a href="https://huggingface.co/aipicasso/cool-japan-diffusion-2-1-1-beta">Cool Japan Diffusion 2.1.1 Beta</a> Stable Diffusion model.<br>
              </p>
              <p>
              sample prompt1 : girl, kimono
              </p>
              <p>
              sample prompt2 : boy, school uniform
              </p>
              <p>
              <a href="https://alfredplpl.hatenablog.com/entry/2023/01/11/182146">日本語の取扱説明書</a>.
              </p>
              Running on {"<b>GPU 🔥</b>" if torch.cuda.is_available() else f"<b>CPU 🥶</b>. For faster inference it is recommended to <b>upgrade to GPU in <a href='https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/settings'>Settings</a></b>"}
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="[your prompt]").style(container=False)
                generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))

              image_out = gr.Image(height=512)
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            with gr.Group():
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
              disable_auto_prompt_correction = gr.Checkbox(label="Disable auto prompt corretion.")

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
                  
    inputs = [prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, disable_auto_prompt_correction]

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
