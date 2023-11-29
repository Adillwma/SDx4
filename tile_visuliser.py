import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class PatchVisualizer:
    def calculate_dynamic_overlap(self, x, window_size, patch_size):
        blocks = int(np.ceil(x / patch_size))
        hangover = (patch_size * blocks) - x
        num_of_overlaps = (blocks * 2) - 2
        overlap = hangover / num_of_overlaps                        # length hanging over = total length of blocks end to end - length of x                     number of overlaps = number of blocks * 2  - 2 as there are 2 overlaps for every block except the first and last which only have 1. if there is only 1 block then there is no overlap
        
        # round down overlap  
        overlap = math.floor(overlap)
        all_but_one_ol = overlap * (num_of_overlaps - 1)
        last_ol = hangover - all_but_one_ol   # to make sure all are ints and there is no remainder

        return overlap, last_ol, blocks

    def visualize_patches(self, image_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        window_size = 128
        min_padding_size = 8                      # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size

        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

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


                # Draw a border around the patchwith the colour genrated from the patch number
                patch_number = c * number_of_windows_in_row + r
                colour = plt.cm.jet(patch_number / (number_of_windows_in_col * number_of_windows_in_row))
                # Convert the float values to integers for the color tuple
                colour_int = tuple(int(x * 255) for x in colour[:-1])
                
                draw.rectangle([x_start_point, y_start_point, x_start_point + window_size, y_start_point + window_size], outline=colour_int)

                # Get the center coordinates of the patch
                center_x = x_start_point + window_size // 2
                center_y = y_start_point + window_size // 2

                # Draw the patch number in large text at the center of the patch
                draw.text((center_x, center_y), str(patch_number), font=font, fill=colour_int, anchor="mm")

        # Display the final image
        image.show()

    

if __name__ == "__main__":
    visualizer = PatchVisualizer()
    visualizer.visualize_patches(r"A:\Users\Ada\Desktop\t1.jpg")
