from upscalar_class import upscaler
from PIL import Image
import os


if __name__ == "__main__":
    # specify local image path
    local_image_paths = [r"A:\Users\Ada\Pictures\EN2wgxqW4AAtwLc.jpg"]  # replace with your actual image path
    output_dir = r"output_data/"
    prompt = ""
    negative_prompt = ""
    num_inference_steps = 10   # Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output
    guidance_scale = 0.5
    patch_size = 120
    padding_size = 8
    use_tensorboard = False
    show_patches = False
    dummy_upscale = False              # For debugging. If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage.

    upscale = upscaler(use_tensorboard=use_tensorboard)
    os.makedirs(output_dir, exist_ok=True)

    for local_image_path in local_image_paths:
            
        upscaled_image = upscale.upscale(local_image_path, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, show_patches, dummy_upscale)
        
        # get input image name
        image_name = os.path.basename(local_image_path)
        image_name = os.path.splitext(image_name)[0]    # remove file extension

        # copy input image to output location 
        low_res_img = Image.open(local_image_path).convert("RGB")
        low_res_img.save(output_dir + image_name + "_Original.png")
        upscaled_image.save(output_dir + image_name + "_Upscaled.png")
