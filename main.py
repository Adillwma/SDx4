from upscalar_class import upscaler
from PIL import Image
import os


if __name__ == "__main__":
    local_image_paths = [r"A:\Users\Ada\Desktop\t3.jpg", r"A:\Users\Ada\Desktop\t4.jpg", r"A:\Users\Ada\Desktop\t5.jpg", r"A:\Users\Ada\Desktop\t6.jpg"]      # specify local image paths
    output_dir = r"output_data/"           # specify output directory
    prompt = ""                            # specify prompt for image upscaling
    negative_prompt = ""                   # specify negative prompt for image upscaling
    num_inference_steps = 25                # Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output
    guidance_scale = 0.5          
    patch_size = 120         
    padding_size = 8               
    callback_steps = 100                    # Number of steps between each callback. Set to 0 to disable callbacks. Callbacks are used to show the progress of the image upscaling.
    blending = False                        # If True, will use soft blend. If False, will use hard blend. 
    blend_mode = "add"            
    
    # Pipeline settings
    xformers = False                       # If True, will use the xformers model. If False, will use the original model.
    cpu_offload = False                    # If True, will use the CPU for the first few inference steps, then switch to the GPU. If False, will use the GPU for all inference steps. If True, will be slower but will use less GPU memory.
    attention_slicing = False              # If True, will use attention slicing. If False, will not use attention slicing. If True, will be slower but will use less GPU memory.

    # Debugging settings
    show_patches = False                   # If True, will show the patches that are being upscaled. If False, will not show the patches that are being upscaled.
    dummy_upscale = False                  # For debugging. If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage.
    use_tensorboard = False                # If True, will use tensorboard to log the progress of the image upscaling. If False, will not use tensorboard. 
    seed = None                            # If None, will use a random seed. If set to a numerical value, will use value as the generator seed.


    upscale = upscaler(xformers, cpu_offload, attention_slicing, seed, use_tensorboard)
    os.makedirs(output_dir, exist_ok=True)

    for local_image_path in local_image_paths:
            
        upscaled_image = upscale.upscale(local_image_path, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, blending, callback_steps, show_patches, dummy_upscale)
        
        # get input image name
        image_name = os.path.basename(local_image_path)
        image_name = os.path.splitext(image_name)[0]    # remove file extension

        # copy input image to output location 
        low_res_img = Image.open(local_image_path).convert("RGB")
        low_res_img.save(output_dir + image_name + "_Original.png")
        upscaled_image.save(output_dir + image_name + "_Upscaled.png")
