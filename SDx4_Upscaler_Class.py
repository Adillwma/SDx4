import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from PyQt6.QtCore import pyqtSignal, QObject
from PIL import Image, ImageChops, ImageEnhance
from diffusers import StableDiffusionUpscalePipeline

### beacuse we are checking if file is greater than patch without padding when a file is exactly of windows size and there is only one pathc then we end up genrating four for no reason
### callback images washed out/odd?
### fix all blending modes. currently only normal works
### clean up face detection code

class SDx4Upscaler(QObject):
    """
    The Upscaler class is responsible for upscaling images using the Stable Diffusion x4 Upscaler model.
    It provides functionality to split an image into patches, upscale each patch, and then combine them back into a single image.
    This class also supports various options such as enabling xFormers for faster processing on CUDA GPUs, CPU offloading for boosted CPU processing, and attention slicing to reduce memory usage.
    It also provides the ability to boost face quality in images using simple Haar Cascade face detection, blend patches together for a smoother result, and show patches for debugging purposes.
    Additonally includes logging and signals that allow easy implementation into GUI applications with progress images. 
    
    __init__ parameters:
    - xformers (bool): If True, enables xFormers for faster processing on CUDA GPUs. If not set then defualts to False.
    - cpu_offload (bool): If True, enables CPU offloading for boosted CPU processing. If not set then defualts to False.
    - attention_slicing (bool): If True, enables attention slicing to reduce memory usage. If not set then defualts to False.
    - seed (int): Seed for the random number generator. If None, a random seed is used. If not set then defualts to None.
    - safety_checker (SafetyChecker): Safety checker to use for the model. If None, the default safety checker is used. If not set then defualts to None.
    - log_level (str): The logging level to use. Possible values are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL". If not set then defualts to "WARNING".
    - log_to_file (bool): If True, logging will go to a file called SDx4Upscaler.log. If False, logging will go to the console. If not set then defualts to False.

    upscale arguments:
    - local_image_path (str): The path to the image to upscale.
    - patch_size (int): The size of the patches to be extracted from the image in pixels. If not set then defualts to 120.
    - padding_size (int): Pixels of padding on right and bottom sides of the patches. If not set then defualts to 8.
    - num_inference_steps (int): The number of inference steps to use. Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output. If not set then defualts to 10.
    - guidance_scale (float): The guidance scale to use. If not set then defualts to 0.5.
    - prompt (str): The prompt to use. If not set then defualts to "".
    - negative_prompt (str): The negative prompt to use. If not set then defualts to "".
    - boost_face_quality (bool): If True, boosts face quality in images. If not set then defualts to False.
    - blending (bool): If True, blends patches together for a smoother result. If not set then defualts to False.
    - callback_steps (int): The number of callback steps to use. If not set then defualts to 1.
    - show_patches (bool): If True, shows patches for debugging purposes. If not set then defualts to False.
    - dummy_upscale (bool): If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage. If not set then defualts to False.

    upscale returns:
    - upscaled_image (Image): The upscaled image.
    
    QSignals:
    - processing_position_signal: Emits (current image, current tile, current iteration) during the upscaling process for external tracking or visulisation to users
    - callback_signal: Emits (current image, current tile, current iteration) at set interval of inference iterations, defined by 'callback_steps' argument, during the upscaling process for external tracking or visulisation to users of the upsclae progress of a tile as the inference progresses
    - tile_complete_signal: Emits (upscaled_tile_image, current_tile_number, local_image_path) at set interval of inference iterations, defined by 'callback_steps' argument, during the upscaling process for showing the user a aniamtion or preview of the tiles being added to the final image as the inference progresses
    """
    processing_position_signal = pyqtSignal(int, int, int, int)     # Emits (current tile, total tiles, current iteration, total iterations) during the upscaling process for external tracking or visulisation to users 
    callback_signal = pyqtSignal(object, int)             # Emits (callback_tile_preview, current_tile_number, local_image_path) during the upscaling process for external tracking or visulisation to users
    tile_complete_signal = pyqtSignal(object, int)        # Emits (upscaled_tile_image, current_tile_number, local_image_path) number of the tile that has just been upscaled

    def __init__(self, xformers=False, cpu_offload=False, attention_slicing=False, seed=None, safety_checker=None, log_level="DEBUG", log_to_file=False):
        """
        possible loging levels: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        logging will go to console if log_to_file is False, otherwise it will go to a file called SDx4Upscaler.log
        """
        super().__init__()  # Call the __init__ method of the parent class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float32, safety_checker=safety_checker)   #, local_files_only=True

        #self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(r"A:\Users\Ada\GitHub\AI_Image_Upscale_Windows\App_Data\models--stabilityai--stable-diffusion-x4-upscaler\snapshots\572c99286543a273bfd17fac263db5a77be12c4c", generator=self.generator, torch_dtype=torch.float32, safety_checker=safety_checker)   #, local_files_only=True
        self.pipeline = self.pipeline.to(self.device)   
        self.transform = transforms.ToTensor()

        if xformers and self.device == "cuda":
            self.pipeline.enable_xformers_memory_efficient_attention()
        if cpu_offload and self.device == "cuda":
            self.pipeline.enable_sequential_cpu_offload()
        if attention_slicing and self.device == "cuda" and not xformers:
            self.pipeline.enable_attention_slicing()

        self.use_tqdm = False  # change back to true if used as a standalone class?!!!!
        self.initialise_logging(log_level, log_to_file, xformers, cpu_offload, attention_slicing)

    def initialise_logging(self, log_level, log_to_file, xformers, cpu_offload, attention_slicing):
        """
        Initialises the logging system.

        arguments:
            - log_level (str): The logging level to use. Possible values are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            - log_to_file (bool): If True, logging will go to a file called SDx4Upscaler.log. If False, logging will go to the console.
        """
        # Create a logger for this class
        self.logger = logging.getLogger('SDx4Upscaler')

        # Set the log level based on user input
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        self.logger.setLevel(numeric_level)

        if numeric_level > 20: # 10 is debug. 20 is INFO, 30 is WARNING, 40 is ERROR, 50 is CRITICAL
            self.pipeline.set_progress_bar_config(disable=True)
            self.use_tqdm = False

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')

        # Create a handler based on the user's preference
        if log_to_file:
            file_handler = logging.FileHandler(r'App_Data/SDx4Upscaler.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Use the logger to emit log messages
        #self.logger.debug('Debug message')
        #self.logger.info('Info message')
        #self.logger.warning('Warning message')
        #self.logger.error('Error message')
        #self.logger.critical('Critical message')

        self.logger.debug('Logging Inititalised')
        self.logger.debug('SDx4Upscaler Inititalised')
        self.logger.debug(f'Inference Compute Device: {self.device}')
        self.logger.debug(f"Pipeline Settings - xFormers: {xformers}, CPU Offload: {cpu_offload}, Attention Slicing: {attention_slicing}")

    def upscale(self, local_image_path, patch_size=120, padding_size=8, num_inference_steps=10, guidance_scale=0.5, prompt="", negative_prompt="", boost_face_quality=False, blending=True, callback_steps=1, show_patches=False, dummy_upscale=False):
        """
        Upscales an image using the Stable Diffusion x4 Upscaler model.     
         
        arguments:
            - local_image_path (str): The path to the image to upscale.
            - patch_size (int): The size of the patches to be extracted from the image in pixels. If not set then defualts to 120.
            - padding_size (int): Pixels of padding on right and bottom sides of the patches. If not set then defualts to 8.
            - num_inference_steps (int): The number of inference steps to use. Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output. If not set then defualts to 10.
            - guidance_scale (float): The guidance scale to use. If not set then defualts to 0.5.
            - prompt (str): The prompt to use. If not set then defualts to "".
            - negative_prompt (str): The negative prompt to use. If not set then defualts to "".
            - boost_face_quality (bool): If True, boosts face quality in images. If not set then defualts to False.
            - blending (bool): If True, blends patches together for a smoother result. If not set then defualts to False.
            - callback_steps (int): The number of callback steps to use. If not set then defualts to 1.
            - show_patches (bool): If True, shows patches for debugging purposes. If not set then defualts to False.
            - dummy_upscale (bool): If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage. If not set then defualts to False.

        returns:
            - upscaled_image (Image): The upscaled image.
        
        """
        
        self.face_modifier = 2  #face_modifier move to user input
        self.callback_steps = callback_steps
        self.filepath = local_image_path
        low_res_img = Image.open(local_image_path).convert("RGB")

        # add face check here 
        if boost_face_quality:
            # check image for faces
            check_for_face_result, faces, facedetection_boundingboxes_debugimage = self.check_for_faces(low_res_img)

            if check_for_face_result:
                # increase num_inference_steps
                self.num_inference_steps_used = num_inference_steps * self.face_modifier
                self.logger.info(f"increasing num_inference_steps to: {num_inference_steps}")
            
            else:
                self.num_inference_steps_used = num_inference_steps
                self.logger.info(f"no faces detected, using num_inference_steps: {num_inference_steps}")

        else:
            check_for_face_result = False
            faces = []
            facedetection_boundingboxes_debugimage = None
            self.num_inference_steps_used = num_inference_steps




        if low_res_img.size[0] <= (patch_size + padding_size) and low_res_img.size[1] <= (patch_size + padding_size):   # check if image is smaller than window size in whcxh case just upscale it
            # if image is smaller than patch size then just upscale it
            upscaled_image = self.pipeline(prompt=prompt, 
                                            image=low_res_img, 
                                            num_inference_steps=self.num_inference_steps_used, 
                                            guidance_scale=guidance_scale, 
                                            negative_prompt=negative_prompt).images[0]
                                            #callback=self.callback, 
                                            #callback_steps=1).images[0]   

        else:
            #  Split the input image into patches of shape (patch_size, patch_size), if there are patches of the image which do not fit into the patch grid, flip the image and continue it from the edge of the image to fill the patch grid.  
            patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, x_overlap, y_overlap, x_last_overlap, y_last_overlap, face_in_patch_list = self.split_image_into_patches(low_res_img, boost_face_quality, check_for_face_result, faces)
            
            self.number_of_patches = len(patches)
            self.logger.info(f"number of patches to process: {self.number_of_patches}")

            upscaled_patches = []

            if self.use_tqdm:
                iterator = tqdm(enumerate(patches), total=len(patches), desc="Upscaling patches", leave=True, unit="patch", colour='green')
            else:
                iterator = enumerate(patches)

            for patch_num, patch in iterator:
                self.patch_num = patch_num

                if dummy_upscale:
                    # strecth the patch to 4x its size
                    upscaled_patch = patch.resize(((patch_size + padding_size) * 4, (patch_size + padding_size) * 4), Image.BICUBIC)
                
                else:
                    if boost_face_quality and face_in_patch_list[patch_num]:
                        self.num_inference_steps_used = num_inference_steps * self.face_modifier
                        self.logger.debug(f"face detected in patch, increasing num_inference_steps to: {self.num_inference_steps_used}")
                    else:
                        self.num_inference_steps_used = num_inference_steps

                    upscaled_patch = self.pipeline(prompt=prompt, 
                                                    image=patch, 
                                                    num_inference_steps=self.num_inference_steps_used, 
                                                    guidance_scale=guidance_scale, 
                                                    negative_prompt=negative_prompt,
                                                    callback=self.callback, 
                                                    callback_steps=1).images[0]   
                    

                upscaled_patches.append(upscaled_patch)
                self.tile_complete_signal.emit(upscaled_patch, patch_num)

            if show_patches:
                self.visualize_patches(patches, number_of_windows_in_row, number_of_windows_in_col)
                self.visualize_patches(upscaled_patches, number_of_windows_in_row, number_of_windows_in_col)

            scaling_factor = upscaled_patches[0].size[0] / patches[0].size[0]   # calculate the scaling factor from the size of the first upscaled patch and the first patch
            upscaled_image = self.reconstruct_from_patches(upscaled_patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, scaling_factor, low_res_img.size, blending)
       
        self.logger.debug("Upscale successful")

        return upscaled_image

    def split_image_into_patches(self, image, boost_face_quality, check_for_face_result, faces):
        # rather than the following tow being set directly there shoudl always be a sum of 128 and then the padding size should be chosen which will then take away from that 128 leaving the patch_width as the remainder
        window_size = 128
        min_padding_size = 8                      # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size

        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

        patches = []
        face_in_patch_list = []
        for c in range(0, number_of_windows_in_col):
            for r in range(0, number_of_windows_in_row):
                if r == number_of_windows_in_row - 1:
                    x_start_point = (r * window_size) - (r * x_last_overlap)
                else:
                    x_start_point = (r * window_size) - (r * x_overlap)

                if c == number_of_windows_in_col - 1:
                    y_start_point = (c * window_size) - (c * y_last_overlap)
                else:
                    y_start_point = (c * window_size) - (c * y_overlap)

                # Crop the patch from the image
                patch = image.crop((x_start_point, y_start_point, x_start_point + window_size, y_start_point + window_size))

                # Append the patch to the list
                patches.append(patch)

                if boost_face_quality and check_for_face_result:
                    # check if patch contains face
                    patch_contains_face = False
                    for face in faces:
                        if (x_start_point > face[0] and x_start_point < face[0] + face[2]) or (x_start_point + window_size > face[0] and x_start_point + window_size < face[0] + face[2]):
                            if (y_start_point > face[1] and y_start_point < face[1] + face[3]) or (y_start_point + window_size > face[1] and y_start_point + window_size < face[1] + face[3]):
                                patch_contains_face = True
                                break

                    if patch_contains_face:
                        face_in_patch_list.append(True)
                    else:
                        face_in_patch_list.append(False)

                else:
                    face_in_patch_list.append(False)

        return patches, number_of_windows_in_row, number_of_windows_in_col, patch_size, x_overlap, y_overlap, x_last_overlap, y_last_overlap, face_in_patch_list

    def check_for_faces(self, image):
        """
        Uses Haar Cascade face detection to check if a face is present in the image tile, and returns a boolean value, also returns the image with bounding boxes drawn around the detected faces for debugging.
        
        Arguments:
            image: numpy array
                The image to check for faces.

        """
        debug_image_with_bounding_boxes = image.copy()

        # make sure image is in form of numpy array if not conver to numpy array
        if not isinstance(debug_image_with_bounding_boxes, np.ndarray):
            debug_image_with_bounding_boxes = np.array(debug_image_with_bounding_boxes)

        # Load the pre-trained Haar Cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale
        gray = cv2.cvtColor(debug_image_with_bounding_boxes, cv2.COLOR_RGB2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

        number_of_faces = len(faces)

        if number_of_faces > 0:
            result = True
        else:
            result = False
        
        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(debug_image_with_bounding_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return result, faces, debug_image_with_bounding_boxes

    def calculate_dynamic_overlap(self, x, window_size, patch_size):
        blocks = int(np.ceil(x / patch_size))
        hangover = (patch_size * blocks) - x
        num_of_overlaps = blocks - 1
        overlap = hangover / num_of_overlaps                        # length hanging over = total length of blocks end to end - length of x                     number of overlaps = number of blocks * 2  - 2 as there are 2 overlaps for every block except the first and last which only have 1. if there is only 1 block then there is no overlap
        
        # round down overlap  
        overlap = np.floor(overlap)
        all_but_one_ol = overlap * (num_of_overlaps - 1)
        last_ol = hangover - all_but_one_ol   # to make sure all are ints and there is no remainder

        overlap = overlap + (window_size - patch_size)
        last_ol = last_ol + (window_size - patch_size)

        return overlap, last_ol, blocks

    def blend_images(self, background, overlay, position, blending_mode="normal"):

        # Calculate the alpha value based on the overlap
        alpha = overlay.convert('L')  # Convert to grayscale
        alpha = ImageEnhance.Brightness(alpha).enhance(0.5)  # Adjust the brightness as needed

        # Apply blending mode based on user selection
        if blending_mode == "normal":
            blended_image = overlay
        elif blending_mode == "add":
            blended_image = ImageChops.add(background, overlay, scale=2.0)
        elif blending_mode == "multiply":
            blended_image = ImageChops.multiply(background, overlay)
        elif blending_mode == "screen":
            blended_image = ImageChops.screen(background, overlay)
        elif blending_mode == "overlay":
            blended_image = ImageChops.overlay(background, overlay)
        elif blending_mode == "soft_light":
            blended_image = ImageChops.soft_light(background, overlay)
        elif blending_mode == "hard_light":
            blended_image = ImageChops.hard_light(background, overlay)
        elif blending_mode == "darken":
            blended_image = ImageChops.darker(background, overlay)
        elif blending_mode == "lighten":
            blended_image = ImageChops.lighter(background, overlay)
        elif blending_mode == "difference":
            blended_image = ImageChops.difference(background, overlay)
        elif blending_mode == "exclusion":
            blended_image = ImageChops.invert(ImageChops.difference(background, overlay))

        # Paste the blended image onto the result image
        background.paste(blended_image, position)

    def reconstruct_from_patches(self, patches, number_of_patches_in_row, number_of_patches_in_col, patch_size, scaling_factor, original_image_shape, blending):
        scaling_factor = int(scaling_factor)
        window_size = 128 * scaling_factor
        min_padding_size = 8 * scaling_factor                                # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size                          # Size of the patches to be extracted from the image in pixels

        # calculate the size of the reconstructed image
        width = original_image_shape[0] * scaling_factor
        height = original_image_shape[1] * scaling_factor

        # Create a new image with the same mode and size as the original image
        reconstructed_image = Image.new(mode='RGB', size=[int(width), int(height)])

        x_overlap, x_last_overlap, number_of_patches_in_row = self.calculate_dynamic_overlap(width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_patches_in_col = self.calculate_dynamic_overlap(height, window_size, patch_size)

        # Paste each patch onto the result image
        for c in range(number_of_patches_in_col):
            for r in range(number_of_patches_in_row):
                if r == number_of_patches_in_row - 1:
                    x_start_point = (r * window_size) - (r * x_last_overlap)
                else:
                    x_start_point = (r * window_size) - (r * x_overlap)
                    
                if c == number_of_patches_in_col - 1:
                    y_start_point = (c * window_size) - (c * y_last_overlap)
                else:
                    y_start_point = (c * window_size) - (c * y_overlap)

                if blending:
                    # Paste the patch onto the result image with blending
                    self.blend_images(reconstructed_image, patches[c * number_of_patches_in_row + r], (int(x_start_point), int(y_start_point)))
                else:
                    # Paste the patch onto the result image
                    reconstructed_image.paste(patches[c * number_of_patches_in_row + r], (int(x_start_point), int(y_start_point)))

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

    def callback(self, iter, t, latents):

        self.processing_position_signal.emit(self.patch_num, self.number_of_patches, iter, self.num_inference_steps_used)

        if iter % self.callback_steps == 0:
            # convert latents to image
            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                image = self.pipeline.vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)

                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16?
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                # convert to PIL Images
                image = self.pipeline.numpy_to_pil(image)

                for i, img in enumerate(image):

                    # report that a new image is ready as a signal so outside class can execute an event 
                    self.callback_signal.emit(img, self.patch_num)

# Demo usage of the SDx4Upscaler class
if __name__ == "__main__":

    # specify local image paths
    local_image_paths = [r"A:\Users\Ada\Desktop\funkyspace.png"] # replace with your actual image path
    prompt = ""
    negative_prompt = ""
    num_inference_steps = 3   # Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output
    guidance_scale = 0.5
    patch_size = 120
    padding_size = 8
    boost_face_quality = False
    callback_steps = 1
    blending = False
    show_patches = False
    dummy_upscale = False              # For debugging. If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage.

    # Create an instance of the SDx4Upscaler class
    upscale = SDx4Upscaler()

    # Implemnt batch processing
    for local_image_path in local_image_paths:
        upscaled_image = upscale.upscale(local_image_path, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, boost_face_quality, blending, callback_steps, show_patches, dummy_upscale)
    
        # copy input image to output location 
        low_res_img = Image.open(local_image_path).convert("RGB")
        low_res_img.save("TEST-low_res.png")
        upscaled_image.save("TEST-upscaled.png")
