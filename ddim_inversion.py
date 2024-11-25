from typing import List

import numpy as np
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler

from attention_control import AttentionStore
import ptp_utils

NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 5.5
START_STEP = 0

attention_maps_gt = []

# Useful function for later
def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = 'Beautiful DSLR Photograph of a penguin on the beach, golden hour'
negative_prompt = 'blurry, ugly, stock photo'
# im = pipe(prompt, negative_prompt=negative_prompt).images[0]
# im.resize((256, 256))
# im.show()
tokenizer = pipe.tokenizer


@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        # 注意力记录
        # latents = controller.step_callback(latents)
        #
        # show_cross_attention(controller, 16, ["up", "down"], step_idx=i)
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue

        t = timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)
        # TODO
        latents = controller.step_callback(latents)
        show_cross_attention(controller, 16, ["up", "down"], step_idx=i)
    return torch.cat(intermediate_latents)


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    # from_where表示从哪些层获取注意力矩阵
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len([input_image_prompt]), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0] # Average over the number of layers
    return out.cpu()

def calculate_similarity(interpolated_att, start_att):
    interpolated_att_normalized = interpolated_att / 255.0
    start_att_normalized = start_att / 255.0
    interpolated_att_flat = interpolated_att_normalized.flatten()
    start_att_flat = start_att_normalized.flatten()
    dot_product = np.dot(interpolated_att_flat, start_att_flat)
    norm1 = np.linalg.norm(interpolated_att_flat)
    norm2 = np.linalg.norm(start_att_flat)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, step_idx=1):
    tokens = tokenizer.encode(input_image_prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)  # attention_maps.shape = (16, 16, 77)
    every_step_similarity=[]
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]  # (16, 16)
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))

        if step_idx == 1:
            attention_maps_gt.append(image)
        similarity = calculate_similarity(image, attention_maps_gt[i])
        every_step_similarity.append(similarity)
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0), step_idx=step_idx)
    ptp_utils.view_similarity(every_step_similarity, step_idx=step_idx, tokens=tokens)

def edit(controller, input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5):
    with torch.no_grad():
        latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
    l = 0.18215 * latent.latent_dist.sample()
    ptp_utils.register_attention_control(pipe, controller)
    # (1, 4, 64, 64)
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)
    # (48, 4, 64, 64)

    final_im = sample(edit_prompt, start_latents=inverted_latents[-(start_step+1)][None], # 这里的[None]是为了增加一个batch维度
                      start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale)[0]
    return final_im
controller = AttentionStore()
# input_image = load_image('data/gnochi_mirror.jpeg', size=(512, 512))
# input_image_prompt = "a cat sitting next to a mirror"
# edit_prompt = "a tiger sitting next to a mirror"

input_image = load_image('data/pexels-photo-8306128.jpeg', size=(512, 512))
input_image_prompt = "a photo of dog"
edit_prompt = "a photo of cat"

img = edit(controller, input_image, input_image_prompt, edit_prompt, num_steps=NUM_DDIM_STEPS, start_step=START_STEP, guidance_scale=GUIDANCE_SCALE)
img.save("results/result.png")
# show_cross_attention(controller, 16, ["up", "down"])





# import matplotlib.pyplot as plt
#
# # 每隔10步选择的索引
# indices = list(range(0, 48, 12))
# num_images = len(indices)
#
# # 设置图像网格的大小：2行，len(indices)列
# fig, axes = plt.subplots(1, num_images, figsize=(20, 5))  # 可调整figsize以改变图像大小
#
# # 循环每个索引，生成并显示图像
# for i, idx in enumerate(indices):
#     # 运行生成代码，得到x_t和image
#     every_step_img = edit(controller, input_image, input_image_prompt, edit_prompt, num_steps=NUM_DDIM_STEPS,
#                start_step=idx, guidance_scale=GUIDANCE_SCALE)
#
#     # 在第1行显示x_t
#     axes[i].imshow(every_step_img)  # 显示x_t
#     axes[i].axis("off")  # 隐藏坐标轴
#     axes[i].set_title(f"Step {idx}")  # 设置标题显示步骤
#
#
# # 调整子图间的间距
# plt.tight_layout()
# plt.savefig('results/every_step_images.png')