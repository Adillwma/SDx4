from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageChops, ImageEnhance

### Fix seams by using a stride less than the patch size so that patches overlap and then cropping the overlap 'bleed' areas when recombining.

### invcesitgate fading the padding areas together vs hard cut

### Fix balck bar on bottom due to padding and the fact that the last patch is not the same size as the others

class upscaler():
    def __init__(self, xformers=False, cpu_offload=False, attention_slicing=False, seed=None, use_tensorboard=False):
        self.generator = torch.manual_seed(seed) if seed else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", generator=self.generator, torch_dtype=torch.float32)   #, local_files_only=True
        self.pipeline = self.pipeline.to(self.device)   

        if xformers:
            self.pipeline.enable_xformers_memory_efficient_attention()
        if cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        if attention_slicing:
            self.pipeline.enable_attention_slicing()

        self.transform = transforms.ToTensor()

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.log_dir = "tensorboard_logs"
            os.makedirs(self.log_dir, exist_ok=True)

    def callback(self, iter, t, latents):
        # convert latents to image
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.pipeline.vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # convert to PIL Images
            image = self.pipeline.numpy_to_pil(image)

            # do something with the Images
            for i, img in enumerate(image):
                img.save(f"iter_{iter}_img{i}.png")

    def split_image_into_patchesOLD(self, image):
        # rather than the following tow being set directly there shoudl always be a sum of 128 and then the padding size should be chosen which will then take away from that 128 leaving the patch_width as the remainder
        padding_size = 8                      # Pixels of padding on right and bottom sides of the patches
        patch_size = 128 - padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size
    
        number_of_patches_in_row = np.ceil(input_image_width / patch_size).astype(int)
        number_of_patches_in_col = np.ceil(input_image_height / patch_size).astype(int)
        print(f"Number of patches in row: {number_of_patches_in_row}")
        print(f"Number of patches in col: {number_of_patches_in_col}")
        print(f"Patch size: {patch_size}")

        patches = []
        for y in range(0, input_image_height, patch_size):
            for x in range(0, input_image_width, patch_size):
                # Crop the patch from the image
                patch = image.crop((x, y, x + patch_size + padding_size, y + patch_size + padding_size))
                print(f"Patch size: {patch.size}")
                # Append the patch to the list
                patches.append(patch)

        return patches, number_of_patches_in_row, number_of_patches_in_col, patch_size

    def calculate_dynamic_overlap(self, x, window_size, patch_size):
        blocks = int(np.ceil(x / patch_size))
        hangover = (window_size * blocks) - x
        num_of_overlaps = (blocks * 2) - 2
        overlap = hangover / num_of_overlaps                        # length hanging over = total length of blocks end to end - length of x                     number of overlaps = number of blocks * 2  - 2 as there are 2 overlaps for every block except the first and last which only have 1. if there is only 1 block then there is no overlap
        
        # round down overlap  
        overlap = math.floor(overlap)

        all_but_one_ol = overlap * (num_of_overlaps - 1)
        last_ol = hangover - all_but_one_ol   # to make sure all are ints and there is no remainder

        print("overlap: ", overlap)
        print("last overlap: ", last_ol)
        print("all but one overlap: ", all_but_one_ol)

        return overlap, last_ol, blocks

    def split_image_into_patches(self, image):
        # rather than the following tow being set directly there shoudl always be a sum of 128 and then the padding size should be chosen which will then take away from that 128 leaving the patch_width as the remainder
        window_size = 128
        min_padding_size = 8                      # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size
        
        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

        patches = []
        for c in range(0, number_of_windows_in_col):
            for r in range(0, number_of_windows_in_row):
                if r == number_of_windows_in_row - 1:
                    x_start_point = (r * window_size) - (r * x_last_overlap * 2)
                else:
                    x_start_point = (r * window_size) - (r * x_overlap * 2)

                if c == number_of_windows_in_col - 1:
                    y_start_point = (c * window_size) - (c * y_last_overlap * 2)
                else:
                    y_start_point = (c * window_size) - (c * y_overlap * 2)




                # Crop the patch from the image
                patch = image.crop((x_start_point, y_start_point, x_start_point + window_size, y_start_point + window_size))

                # Append the patch to the list
                patches.append(patch)

        return patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, x_overlap, y_overlap, x_last_overlap, y_last_overlap
        
    def reconstruct_from_patches(self, patches, number_of_patches_in_row, number_of_patches_in_col, patch_size, scaling_factor, x_overlap, y_overlap, x_last_overlap, y_last_overlap, original_image_shape):
        scaling_factor = int(scaling_factor)
        window_size = 128 * scaling_factor
        min_padding_size = 8 * scaling_factor                                # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size                          # Size of the patches to be extracted from the image in pixels

        # calculate the size of the reconstructed image
        width = original_image_shape[0] * scaling_factor
        height = original_image_shape[1] * scaling_factor

        # Create a new image with the same mode and size as the original image
        reconstructed_image = Image.new(mode='RGB', size=[int(width), int(height)])

        # Paste each patch onto the result image
        for c in range(number_of_patches_in_col):
            for r in range(number_of_patches_in_row):
                if r == number_of_patches_in_row - 1:
                    x_start_point = (r * window_size) - (r * x_last_overlap  * scaling_factor * 2)
                else:
                    x_start_point = (r * window_size) - (r * x_overlap * scaling_factor * 2)
                if c == number_of_patches_in_col - 1:
                    y_start_point = (c * window_size) - (c * y_last_overlap * scaling_factor * 2)
                else:
                    y_start_point = (c * window_size) - (c * y_overlap * scaling_factor * 2)

                # Paste the patch onto the result image
                reconstructed_image.paste(patches[c * number_of_patches_in_row + r], (x_start_point, y_start_point))

        return reconstructed_image

    def blend_images(self, background, overlay, position):
        # Calculate the alpha value based on the overlap
        alpha = overlay.convert('L')  # Convert to grayscale
        alpha = ImageEnhance.Brightness(alpha).enhance(0.5)  # Adjust the brightness as needed

        # Paste the overlay onto the background with blending
        background.paste(overlay, position, mask=alpha)

    def reconstruct_from_patches_w_blending(self, patches, number_of_patches_in_row, number_of_patches_in_col, patch_size, scaling_factor, x_overlap, y_overlap, x_last_overlap, y_last_overlap, original_image_shape, blending=True):
        scaling_factor = int(scaling_factor)
        window_size = 128 * scaling_factor
        min_padding_size = 8 * scaling_factor                                # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size                          # Size of the patches to be extracted from the image in pixels

        # calculate the size of the reconstructed image
        width = original_image_shape[0] * scaling_factor
        height = original_image_shape[1] * scaling_factor

        # Create a new image with the same mode and size as the original image
        reconstructed_image = Image.new(mode='RGB', size=[int(width), int(height)])

        # Paste each patch onto the result image
        for c in range(number_of_patches_in_col):
            for r in range(number_of_patches_in_row):
                if r == number_of_patches_in_row - 1:
                    x_start_point = (r * window_size) - (r * x_last_overlap  * scaling_factor * 2)
                else:
                    x_start_point = (r * window_size) - (r * x_overlap * scaling_factor * 2)
                if c == number_of_patches_in_col - 1:
                    y_start_point = (c * window_size) - (c * y_last_overlap * scaling_factor * 2)
                else:
                    y_start_point = (c * window_size) - (c * y_overlap * scaling_factor * 2)

                if blending:
                    # Paste the patch onto the result image with blending
                    self.blend_images(reconstructed_image, patches[c * number_of_patches_in_row + r], (x_start_point, y_start_point))
                else:
                    # Paste the patch onto the result image
                    reconstructed_image.paste(patches[c * number_of_patches_in_row + r], (x_start_point, y_start_point))

        return reconstructed_image

    def visualize_patches(self, patches, number_of_windows_in_row, number_of_windows_in_col):
        # Calculate the number of rows and columns for the grid
        num_cols = number_of_windows_in_col
        num_rows = number_of_windows_in_row

        # Create a new figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        # Flatten the axes array if there's only one row or column
        axes = np.array(axes).flatten()

        for i, ax in enumerate(axes):
            # Hide the axes
            ax.axis('off')

            if i < len(patches):
                # Display the patch on the current axis
                ax.imshow(patches[i])
                ax.set_title(f'Patch {i + 1}', fontsize=8)

        plt.show()

    def update_animation(self, upscaled_patches):
        pass

    def upscale(self, local_image_path, patch_size=120, padding_size=8, num_inference_steps=10, guidance_scale=0.5, prompt="", negative_prompt="", callback_steps=1, show_patches=False, dummy_upscale=False):
        # specify local image path
        low_res_img = Image.open(local_image_path).convert("RGB")

        #  Split the input image into patches of shape (patch_size, patch_size), if there are patches of the image which do not fit into the patch grid, flip the image and continue it from the edge of the image to fill the patch grid.  
        patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, x_overlap, y_overlap, x_last_overlap, y_last_overlap = self.split_image_into_patches(low_res_img)

        print("number of patches: ", len(patches))

        upscaled_patches = []
        for i, patch in enumerate(patches):

            if dummy_upscale:
                # strecth the patch to 4x its size
                upscaled_patch = patch.resize(((patch_size + padding_size) * 4, (patch_size + padding_size) * 4), Image.BICUBIC)
            
            else:
                upscaled_patch = self.pipeline(prompt=prompt, 
                                                image=patch, 
                                                num_inference_steps=num_inference_steps, 
                                                guidance_scale=guidance_scale, 
                                                negative_prompt=negative_prompt,
                                                callback=self.callback, 
                                                callback_steps=callback_steps).images[0]   
                

            upscaled_patches.append(upscaled_patch)
            #self.update_animation(upscaled_patches)

        if show_patches:
            self.visualize_patches(patches, number_of_windows_in_row, number_of_windows_in_col)
            self.visualize_patches(upscaled_patches, number_of_windows_in_row, number_of_windows_in_col)

        scaling_factor = upscaled_patches[0].size[0] / patches[0].size[0]   # calculate the scaling factor from the size of the first upscaled patch and the first patch
        upscaled_image = self.reconstruct_from_patches(upscaled_patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, scaling_factor, x_overlap, y_overlap, x_last_overlap, y_last_overlap, low_res_img.size)
        print("Upscale successful")

        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)
            writer.add_image("low_res_image", self.transform(low_res_img))      # Log the low-resolution image to TensorBoard
            writer.add_image("upscaled_image", self.transform(upscaled_image))                   # Log the upscaled image to TensorBoard
            hparams_dict = {
                "num_inference_steps": num_inference_steps,
                "patch_size": patch_size,
                "guidance_scale": guidance_scale,
                "prompt": prompt,
                "negative_prompt": negative_prompt
            }  
            writer.add_hparams(hparams_dict, {})                                                 # Log the hyperparameters to TensorBoard  
            writer.close()                                                                       # Close the TensorBoard writer

        return upscaled_image
    
if __name__ == "__main__":
    # specify local image path
    local_image_path = r"A:\Users\Ada\Pictures\1451500_524782900944582_1282936311_n.jpg"  # replace with your actual image path
    prompt = ""
    negative_prompt = ""
    num_inference_steps = 25   # Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output
    guidance_scale = 0.5
    patch_size = 120
    padding_size = 8
    use_tensorboard = True
    callback_steps = 1
    show_patches = False
    dummy_upscale = False              # For debugging. If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage.

    upscale = upscaler(use_tensorboard=use_tensorboard)
    upscaled_image = upscale.upscale(local_image_path, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, callback_steps, show_patches, dummy_upscale)
    
    # copy input image to output location 
    low_res_img = Image.open(local_image_path).convert("RGB")
    low_res_img.save("low_resP2.png")
    upscaled_image.save("upscaledP2.png")
