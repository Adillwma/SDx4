import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import matplotlib.pyplot as plt
import timeit

num_inference_steps = 5

def version_1():    #  STANDARD VERSION
    # Original code
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    prompt = "a white cat"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=num_inference_steps).images[0]

    return low_res_img, upscaled_image

def version_2():    # WITH NO GRAD
    # Original code
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    prompt = "a white cat"

    with torch.no_grad():
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=num_inference_steps).images[0]

    return low_res_img, upscaled_image

def version_3():    # WITH NO GRAD
    # Original code
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    prompt = "a white cat"

    with torch.no_grad():
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=num_inference_steps).images[0]

    return low_res_img, upscaled_image

def version_4():    # WITH NO GRAD
    # Original code
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    prompt = "a white cat"

    with torch.no_grad():
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=num_inference_steps).images[0]

    return low_res_img, upscaled_image

# Run the versions
#low_res_img_v1, upscaled_img_v1 = version_1()
# low_res_img_v2, upscaled_img_v2 = version_2()  # Uncomment and modify this line for version 2

# Plot the images side by side
#fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#axes[0].imshow(low_res_img_v1)
#axes[0].set_title('Low-Resolution Image (Version 1)')
#axes[1].imshow(upscaled_img_v1)
#axes[1].set_title('Upscaled Image (Version 1)')

# Uncomment the following lines for version 2
# axes[2].imshow(low_res_img_v2)
# axes[2].set_title('Low-Resolution Image (Version 2)')
# axes[3].imshow(upscaled_img_v2)
# axes[3].set_title('Upscaled Image (Version 2)')

plt.show()

# Measure time for version 1
time_v1 = timeit.timeit(version_1, number=2)

# Measure time for version 2
time_v2 = timeit.timeit(version_2, number=2)

# Plot the runtimes
versions = ['Version 1', 'Version 2']
times = [time_v1, time_v2]

plt.bar(versions, times)
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison of Execution Time for Different Versions')
plt.show()
