# Thank AK. https://huggingface.co/spaces/akhaliq/cool-japan-diffusion-2-1-0/blob/main/app.py
from PIL import Image
import torch
import gradio as gr
from transformers import CLIPImageProcessor
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionInpaintPipeline
import sys
import tempfile
import inspect
import diffusers
import sys
from diffusers.schedulers.scheduling_utils import SchedulerMixin


import os
import time
if not os.path.exists('pic'):
    os.mkdir('pic')
module = sys.modules['tempfile']
def fn(): return '.\\pic'


module._get_default_tempdir = fn
sys.modules['tempfile'] = module

schedulers = [(name, obj) for name, obj in inspect.getmembers(
    sys.modules['diffusers']) if inspect.isclass(obj) and issubclass(obj, SchedulerMixin) and name != "KarrasVeScheduler"]
result_paired = list(set([(name, parameter.default) for (_, obj) in schedulers for (name, parameter) in inspect.signature(
    inspect.getattr_static(obj, "__init__")).parameters.items() if ((name != "self") & (name != "kwargs"))]))

result = []
seen_names = set()
for name, size in result_paired:
    if name not in seen_names:
        result.append((name, size))
        seen_names.add(name)
print(result)
to_render = dict()
for name, default in result:
    print(name)
    print(default)
    if (name == "beta_schedule"):
        to_render[name] = (gr.Dropdown(choices=[
            "linear", "scaled_linear", "squaredcos_cap_v2"], value=default, label=name))
    elif (name == "solver_order"):
        to_render[name] = (
            gr.Slider(1, 3, value=default, label=name, visible=False))
    elif (name == "solver_type"):
        to_render[name] = (gr.Dropdown(
            choices=["midpoint", "heun"], value=default, label=name, visible=False))
    elif (name == "algorithm_type"):
        to_render[name] = (gr.Dropdown(
            choices=["dpmsolver++", "dpmsolver"], value=default, label=name, visible=False))
    elif (name == "prediction_type"):
        to_render[name] = (gr.Dropdown(
            choices=["epsilon", "sample", "v_prediction"], value=default, label=name, visible=False))
    elif (type(default) == type(0.5)):
        to_render[name] = (
            gr.Number(value=default, label=name, precision=None, visible=False))
    elif (type(default) == type(1)):
        to_render[name] = (
            gr.Number(value=default, label=name, precision=0, visible=False))
    elif (type(default) == type(True)):
        to_render[name] = (gr.Checkbox(
            value=default, label=name, visible=False))
    elif (type(default) == type("str")):
        to_render[name] = (gr.Text(value=default, label=name, visible=False))
    else:
        to_render[name] = (gr.State(value=None, label=name, visible=False))


class custom_bag:
    current_custom_scheduler = None
    current_filled_settings = None
    pipe_i2i = None
    pipe = None


def fill_defaults(data):
    print(data)
    return_arr = []
    custom_bag.current_filled_settings = dict()
    scheduler = next((obj for name, obj in schedulers if name == data), False)
    for number in to_render:
        if (scheduler):
            custom_bag.current_custom_scheduler = scheduler
            parameter = next((obj for name, obj in inspect.signature(
                inspect.getattr_static(scheduler, "__init__")).parameters.items() if name == number), False)
            if (parameter):
                print(parameter.default)
                return_arr.append(
                    gr.update(visible=True, value=parameter.default))
                custom_bag.current_filled_settings[number] = parameter.default
                continue
        return_arr.append(gr.update(visible=False))

    return return_arr


def set_values(data):
    print(data)
    return_arr = []
    custom_bag.current_filled_settings = dict()
    scheduler = next((obj for name, obj in schedulers if name == data), False)
    for number in to_render:
        if (scheduler):
            custom_bag.current_custom_scheduler = scheduler
            parameter = next((obj for name, obj in inspect.signature(
                inspect.getattr_static(scheduler, "__init__")).parameters.items() if name == number.label), False)
            if (parameter):
                print(parameter.default)
                return_arr.append(
                    gr.update(visible=True))
                custom_bag.current_filled_settings[number.label] = number.value
                continue
        return_arr.append(gr.update(visible=False))

    return return_arr


def error_str(error, title="Error"):
    return f"""#### {title}
            {error}""" if error else ""


model_id = 'aipicasso/cool-japan-diffusion-2-1-1-beta'
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler")
print(scheduler)
feature_extractor = CLIPImageProcessor.from_pretrained(model_id)

custom_bag.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    scheduler=scheduler,
    requires_safety_checker=False,
    safety_checker=None,
    feature_extractor=feature_extractor
)
custom_bag.pipe = StableDiffusionPipeline(**custom_bag.pipe_i2i.components)

if torch.cuda.is_available():
    custom_bag.pipe = custom_bag.pipe.to("cuda")
    custom_bag.pipe_i2i = custom_bag.pipe_i2i.to("cuda")


def inference(data, progress=gr.Progress()):

    if (data[advanced_check]):
        print(custom_bag.current_filled_settings)
        # set_values(scheduler_dropdown)
        kwargs = {inputs.label: data[inputs] for inputs in [*data] if type(inputs.label) is type("str")}
        print(kwargs)
        scheduler = next(
            (obj for name, obj in schedulers if name == kwargs['scheduler']), False)
        
                # Get the list of valid parameters
        valid_params = inspect.signature(scheduler).parameters.keys()

        # Create a copy of the parameters dictionary
        params_copy = kwargs.copy()

        # Iterate over the parameters and remove any that are not in the list of valid parameters
        for key, value in params_copy.items():
            if key not in valid_params:
                kwargs.pop(key)

        # Print the remaining parameters
        print(kwargs)
        custom_bag.pipe.scheduler = scheduler(
            **kwargs)
        custom_bag.pipe_i2i.scheduler = scheduler(
            **kwargs)

    generator = torch.Generator('cuda').manual_seed(
        data[seed]) if data[seed] != 0 else None

    if (not data[disable_auto_prompt_correction]):
        data[prompt], data[neg_prompt] = auto_prompt_correction(
            data[prompt], data[neg_prompt], data[cool_japan_type])

    if (data[image_size] == "Portrait"):
        height = 768
        width = 576
    elif (data[image_size] == "Landscape"):
        height = 576
        width = 768
    elif (data[image_size] == "Square"):
        height = 512
        width = 512
    else:
        height = 768
        width = 768
    try:
        if data[image] is not None:
            return img_to_img(data[prompt], data[neg_prompt], data[image], data[strength], data[guidance], data[steps], width, height, generator, progress), None
        else:
            return txt_to_img(data[prompt], data[neg_prompt], data[guidance], data[steps], width, height, generator, progress), None
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
        image = custom_bag.pipe.decode_latents(latents)
        # prog_holder.images.append(image[0])
        prog((step % steps if steps != 0 else 0, steps))
        prog_holder.current_img = image[0]
    return collect


def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator, progress):
    print(custom_bag.pipe.scheduler)
    result = custom_bag.pipe(
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
    print(custom_bag.pipe_i2i.scheduler)
    ratio = min(height / img.height, width / img.width)
    img = img.resize(
        (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = custom_bag.pipe_i2i(
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
                image = gr.Image(tool="editor", label="Image", type="pil")
                strength = gr.Slider(
                    label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Tab('Advanced Options'):
            with gr.Group():
                advanced_check = gr.Checkbox(
                    value=False, label="Enable custom scheduler")
                scheduler_dropdown = gr.Dropdown(
                    [name for name, _ in schedulers], value=schedulers[0][0], visible=False, interactive=True, label="scheduler")
                load_default_button = gr.Button("Hide Invalid", visible=False)
                with gr.Group():
                    [number.render() for number in to_render.values()]

    load_default_button.click(fill_defaults, scheduler_dropdown, [
                              things for things in to_render.values()])

    all_inputs = [scheduler_dropdown] + \
        [things for things in to_render.values()]
    advanced_check.change(lambda val: [gr.update(
        visible=val, interactive=val) for _ in all_inputs]+[gr.update(
            visible=val)], advanced_check, all_inputs + [load_default_button])

    inputs = {prompt, guidance, steps, image_size, seed, image, strength, neg_prompt,
              cool_japan_type, disable_auto_prompt_correction, advanced_check, scheduler_dropdown}
    inputs = {*inputs, *to_render.values()}
    outputs = [image_out, error_output]

    prompt.submit(inference, inputs=inputs,
                  outputs=outputs, show_progress=True)
    generate.click(inference, inputs=inputs,
                   outputs=outputs, show_progress=True)

    demo.queue(status_update_rate=1).launch(share=False)
