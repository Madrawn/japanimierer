# Thank AK. https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/blob/main/app.py
from PIL import Image
import torch
import gradio as gr
from transformers import CLIPImageProcessor
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionInpaintPipeline
import sys
import tempfile
import os
import time
if not os.path.exists('pic'):
    os.mkdir('pic')
module = sys.modules['tempfile']
def fn(): return '.\\pic'


module._get_default_tempdir = fn
sys.modules['tempfile'] = module


model_id = 'aipicasso/cool-japan-diffusion-2-1-1-beta'

scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
feature_extractor = CLIPImageProcessor.from_pretrained(model_id)

pipe = StableDiffusionPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  requires_safety_checker=False,
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
            {error}""" if error else ""


def inference(prompt, guidance, steps, image_size="Square", seed=0, img=None, strength=0.5, neg_prompt="", cool_japan_type="Anime", disable_auto_prompt_correction=False, width_custom=512, height_custom=512, progress=gr.Progress()):

    generator = torch.Generator('cuda').manual_seed(
        seed) if seed != 0 else None

    if (not disable_auto_prompt_correction):
        prompt, neg_prompt = auto_prompt_correction(
            prompt, neg_prompt, cool_japan_type)

    if (image_size == "Portrait"):
        height = 768
        width = 576
    elif (image_size == "Landscape"):
        height = 576
        width = 768
    elif (image_size == "Square"):
        height = 512
        width = 512
    else:
        height = 768
        width = 768
    try:
        if img is not None:
            return img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator, progress), None
        else:
            return txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator, progress), None
    except Exception as e:
        return None, error_str(e)


def auto_prompt_correction(prompt_ui, neg_prompt_ui, cool_japan_type_ui):
    # auto prompt correction
    cool_japan_type = str(cool_japan_type_ui)
    if (cool_japan_type == "Manga"):
        cool_japan_type = "manga, monochrome, white and black manga"
    elif (cool_japan_type == "Game"):
        cool_japan_type = "game"
    else:
        cool_japan_type = "anime"

    prompt = str(prompt_ui)
    neg_prompt = str(neg_prompt_ui)
    prompt = prompt.lower()
    neg_prompt = neg_prompt.lower()
    if (prompt == "" and neg_prompt == ""):
        prompt = f"{cool_japan_type}, a portrait of a girl, 4k, detailed"
        neg_prompt = f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    splited_prompt = prompt.replace(",", " ").replace("_", " ").split(" ")
    splited_prompt = ["a person" if p == "solo" else p for p in splited_prompt]
    splited_prompt = ["girl" if p == "1girl" else p for p in splited_prompt]
    splited_prompt = ["a couple of girls" if p ==
                      "2girls" else p for p in splited_prompt]
    splited_prompt = ["a couple of boys" if p ==
                      "2boys" else p for p in splited_prompt]
    human_words = ["girl", "maid", "maids", "female", "woman", "girls", "a couple of girls",
                   "women", "boy", "boys", "a couple of boys", "male", "man", "men", "guy", "guys"]

    prompt = f"{cool_japan_type}, {prompt}, 4k, detailed"
    neg_prompt = f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    animal_words = ["cat", "dog", "bird"]
    for word in animal_words:
        if (word in splited_prompt):
            prompt = f"{cool_japan_type}, a {word}, 4k, detailed"
            neg_prompt = f"(((deformed))), blurry, ((((bad anatomy)))), {neg_prompt}, bad pupil, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), bad hands, fused fingers, messy drawing, broken legs censor, low quality, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), ((bad eyes)), ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 2d, 3d, cg, text"

    background_words = ["mount fuji", "mt. fuji", "building",
                        "buildings", "tokyo", "kyoto", "nara", "shibuya", "shinjuku"]
    for word in background_words:
        if (word in splited_prompt):
            prompt = f"{cool_japan_type}, shinkai makoto, {word}, 4k, 8k, highly detailed"
            neg_prompt = f"(((deformed))), {neg_prompt}, girl, boy, photo, people, low quality, ui, error, lowres, jpeg artifacts, 2d, 3d, cg, text"

    return prompt, neg_prompt


class prog_holder:
    current_img = None
    images = []


def current_img_get():
    return prog_holder.current_img


def numpy_to_png(arr):
    # get current timestamp
    ts = time.time()

    # create image from numpy array
    arr = (arr * 255).round().astype("uint8")
    img = Image.fromarray(arr, 'RGB')
    # save image to pic folder
    img.save('pic/{}.png'.format(ts))


def imgCollector(prog, steps):
    def collect(step: int, timestep: int, latents: torch.FloatTensor):
        image = pipe.decode_latents(latents)
        # prog_holder.images.append(image[0])
        prog((step, steps))
        prog_holder.current_img = image[0]
    return collect


def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator, progress):
    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
        callback=imgCollector(progress, steps))
    # prog_holder.images.append(result.images[0])
    numpy_to_png(prog_holder.current_img)
    return prog_holder.current_img


def img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator, progress):
    ratio = min(height / img.height, width / img.width)
    img = img.resize(
        (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe_i2i(
        prompt,
        negative_prompt=neg_prompt,
        image=img,
        num_inference_steps=int(steps/strength),
        strength=strength,
        guidance_scale=guidance,
        # width = width,
        # height = height,
        generator=generator,
        callback=imgCollector(progress, steps))
    # prog_holder.images.append(result.images[0])
    numpy_to_png(prog_holder.current_img)

    return prog_holder.current_img


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
              </p>
              Running on {"<b>GPU 🔥</b>" if torch.cuda.is_available() else f"<b>CPU 🥶</b>. For faster inference it is recommended to <b>upgrade to GPU in <a href='https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/settings'>Settings</a></b>"}
            </div>
        """
    )
    with gr.Row():
        with gr.Group():
            with gr.Row():
                cool_japan_type = gr.Radio(["Anime", "Manga", "Game"])
                cool_japan_type.show_label = False
                cool_japan_type.value = "Anime"

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,
                                    placeholder="[your prompt]").style(container=False)
                generate = gr.Button(value="Generate").style(
                    rounded=(False, True, True, False))

            image_out = gr.Image()
        image_progress = gr.Image(
            value=current_img_get, every=0.5, interactive=False)
        
    error_output = gr.Markdown()
    with gr.Row():
        with gr.Tab("Options"):
            with gr.Group():
                neg_prompt = gr.Textbox(
                    label="Negative prompt", placeholder="What to exclude from the image")
                disable_auto_prompt_correction = gr.Checkbox(
                    label="Disable auto prompt corretion.")
                with gr.Row():
                    image_size = gr.Radio(
                        ["Portrait", "Landscape", "Square", "Bigger Square"])
                    image_size.show_label = False
                    image_size.value = "Portrait"
                with gr.Row():
                    guidance = gr.Slider(
                        label="Guidance scale", value=7.5, maximum=300)
                    steps = gr.Slider(label="Steps", value=20,
                                      minimum=2, maximum=300, step=1)

                seed = gr.Slider(
                    0, 2147483647, label='Seed (0 = random)', value=0, step=1)
        with gr.Tab("Image to image"):
            with gr.Group():
                image = gr.ImagePaint(label="Image", type="pil")
                strength = gr.Slider(
                    label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)


    inputs = [prompt, guidance, steps, image_size, seed, image, strength, neg_prompt,
              cool_japan_type, disable_auto_prompt_correction]
    outputs = [image_out, error_output]

    prompt.submit(inference, inputs=inputs,
                  outputs=outputs, show_progress=True)
    generate.click(inference, inputs=inputs,
                   outputs=outputs, show_progress=True)

demo.queue(status_update_rate=1).launch(share=False)
