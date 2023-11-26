from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

#
num_inference_steps = 5 
guidance_scale = 1
prompt = None
negative_prompt = None


# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline = pipeline.to("cpu")   # use CPU for inference

# specify local image path
local_image_path = r"A:\Users\Ada\Pictures\rd.png"  # replace with your actual image path
low_res_img = Image.open(local_image_path).convert("RGB")
#low_res_img = low_res_img.resize((128, 128))
# save image
low_res_img.save("low_res_cat_5steps.png")



upscaled_image = pipeline(prompt=prompt, 
                          image=low_res_img, 
                          num_inference_steps=num_inference_steps, 
                          guidance_scale=guidance_scale, 
                          negative_prompt=negative_prompt,
                          ).images[0]   
upscaled_image.save("upsampled_cat_5steps.png")
