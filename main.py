from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from torch.utils.tensorboard import SummaryWriter
import os

# Create a directory to store TensorBoard logs and saved images
log_dir = "tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)

# Create a TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline = pipeline.to("cpu")   # use CPU for inference

# specify local image path
local_image_path = r"A:\Users\Ada\Pictures\rd.png"  # replace with your actual image path
low_res_img = Image.open(local_image_path).convert("RGB")
#low_res_img = low_res_img.resize((128, 128))
# save image
low_res_img.save(os.path.join(log_dir, "low_res_cat_camelprompt.png"))

# Log the low-resolution image to TensorBoard
writer.add_image("Low Resolution Image", torch.tensor(low_res_img).permute(2, 0, 1).unsqueeze(0))

prompt = "a lady"

# Perform inference
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

# Save the upscaled image
upscaled_image_path = os.path.join(log_dir, "upsampled_cat_camelprompt.png")
upscaled_image.save(upscaled_image_path)

# Log the upscaled image to TensorBoard
writer.add_image("Upscaled Image", torch.tensor(upscaled_image).permute(2, 0, 1).unsqueeze(0))

# Close the TensorBoard writer
writer.close()
