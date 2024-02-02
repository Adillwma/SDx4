import os
import sys
import json
import requests
import numpy as np
from PyQt6 import uic
import matplotlib.pyplot as plt
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtXml import QDomDocument
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QColorDialog, QDialog, QLabel
from PyQt6.QtGui import QPixmap
from PIL.ImageQt import ImageQt

class ImageWidget(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, width):
        if self.pixmap():
            return int(width * (self.pixmap().height() / self.pixmap().width()))

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap():
            return self.pixmap().size()
        return super().minimumSizeHint()
        



## - As building a exe file with no console, transformers library will suffer from an error where sys.stdout and sys.stderr are None
# Below four lines fix this issue by redirecting stdout and stderr to os.devnull as suggested here: https://github.com/huggingface/transformers/issues/24047#issuecomment-1635532509
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

#from customWidgets import CustomToggle, CustomIcon
import resources_rc

# trye to take the signals directly from file_processing_thread.upscaler to the gui clss without going through the file_processing_thread classs, aslo then can xsperate the image complete proress signal freom the other two and put image complete after the image is porcessed in file_processing_thread

#%% Helper Functions
def resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#%% - Backend Program Functions
from SDx4_Upscaler_Class import SDx4Upscaler

class FileHandlingThread(QThread):
    def __init__(self, temp_folder, previous_processing_file_list=None):
        super().__init__()
        self.temp_folder = temp_folder
        self.processing_file_list = [] if previous_processing_file_list is None else previous_processing_file_list

    def add_files(self, file_paths):
        for file_path in file_paths:
            self.add_to_processing_list(file_path)

    def add_folder(self, folder_path):    # update to add mulit folder at once
        for file in os.listdir(folder_path):                          # add full file paths to the processing file list
            self.add_to_processing_list(os.path.join(folder_path, file))

    def remove_file(self, selected_item):
        self.processing_file_list.pop(selected_item)

    def add_to_processing_list(self, file_path):

        # check file size on disk 
        file_size = os.path.getsize(file_path)

        # convert file size from bytes to KB or MB depending on size
        if file_size > 1000000:
            file_size = str(round(file_size / 1000000, 2)) + " MB"
        else:
            file_size = str(round(file_size / 1000, 2)) + " KB"


        # load the image file 
        image = Image.open(file_path).convert("RGB")

        # Create preview image
        preview_image = image.copy()
        preview_image = preview_image.resize((image.width * 4, image.height * 4), Image.NEAREST)   
        preview_image = preview_image.convert("RGBA") # convert to RGBA so that the image is loadable into the Qimage and can be displayed in the GUI  

        # Detemine image resoloutions
        input_res = (image.width, image.height)                              # determine the input resoloution
        output_res = (preview_image.width, preview_image.height)             # determine the output resoloution (same as preview, 4x input res)

        # add to the processing file list
        file_info_dict = {}
        file_info_dict["input_file"] = image
        file_info_dict["input_file_path"] = file_path
        file_info_dict["preview_file"] = preview_image
        file_info_dict["input_res"] = input_res
        file_info_dict["output_res"] = output_res
        file_info_dict["file_size"] = file_size

        self.processing_file_list.append(file_info_dict)

    def get_file_images(self, file_number):
        input_image = self.processing_file_list[file_number]["input_file"]
        preview_image = self.processing_file_list[file_number]["preview_file"]

        return input_image, preview_image

    def get_file_info(self, file_number):
        file_size = self.processing_file_list[file_number]["file_size"]
        input_res = self.processing_file_list[file_number]["input_res"]
        output_res = self.processing_file_list[file_number]["output_res"]

        return file_size, input_res, output_res
    
# File processing thread
class FileProcessingThread(QThread):
    processing_position_signal = pyqtSignal(int, int, int, int, int, int)  # Emits (current image, current tile, current iteration) during the upscaling process for gui progress bars
    patch_preview = pyqtSignal(object, int, int)  # Signal for previewing patches
    tile_complete_signal = pyqtSignal(object, int, int)  # Signal for tile complete image retrival
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self):
        super().__init__()

    def send_patch_preview(self, img_patch, patch_num):
        self.patch_preview.emit(img_patch, patch_num, self.current_image_num)

    def send_tile_complete(self, tile_complete_image, tile_num):
        self.tile_complete_signal.emit(tile_complete_image, tile_num, self.current_image_num)

    def send_progress_update(self, current_tile, total_tiles, current_iteration, total_iterations):
        self.processing_position_signal.emit(self.current_image, self.number_of_images, current_tile, total_tiles, current_iteration, total_iterations)

    def run(self):
        for current_image_num, local_image_path in enumerate(self.local_image_paths):
            self.current_image_num = current_image_num
            self.current_image = current_image_num + 1

            ### missing params i.e blend mode!!!
            upscaled_image = self.upscaler.upscale(local_image_path, 
                                                  self.patch_size, 
                                                  self.padding_size, 
                                                  self.num_inference_steps, 
                                                  self.guidance_scale, 
                                                  self.prompt, 
                                                  self.negative_prompt, 
                                                  self.boost_face_quality, 
                                                  self.blending, 
                                                  self.callback_steps, 
                                                  self.show_patches, 
                                                  self.dummy_upscale)
            
            # get input image name
            image_name = os.path.basename(local_image_path)
            image_name = os.path.splitext(image_name)[0]  # remove file extension

            # copy input image to output location
            low_res_img = Image.open(local_image_path).convert("RGB")
            low_res_img.save(os.path.join(self.output_dir, image_name + "_Original.png"))
            upscaled_image.save(os.path.join(self.output_dir, image_name + "_Upscaled.png"))

        # CLENA UP IMAGES FROM THE TEMP FOLDER HERE TOO OR COPULD KEEP THEM FOR A COMPARISON VIEWER PAGE OR SOMTHING?????
            
        # Emit the finished signal when processing is done
        self.finished.emit()

    
    def initialize_upscale_job(self, processing_file_list, output_dir, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, boost_face_quality, blending, blend_mode, callback_steps, show_patches, dummy_upscale, xformers, cpu_offload, attention_slicing, seed, safety_checker):

        self.local_image_paths = [file_info["input_file_path"] for file_info in processing_file_list]
        self.number_of_images = len(self.local_image_paths)

        self.upscaler = SDx4Upscaler(xformers, cpu_offload, attention_slicing, seed, safety_checker)
        self.upscaler.callback_signal.connect(self.send_patch_preview)
        self.upscaler.tile_complete_signal.connect(self.send_tile_complete)
        self.upscaler.processing_position_signal.connect(self.send_progress_update)
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.patch_size = patch_size
        self.padding_size = padding_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.boost_face_quality = boost_face_quality
        self.blending = blending
        self.callback_steps = callback_steps
        self.show_patches = show_patches
        self.dummy_upscale = dummy_upscale

# Upscale preview thread
class UpscalePreviewThread(QThread):
    preview_update_signal = pyqtSignal(object)  # Emits (current image, current tile, current iteration) during the upscaling process for gui progress bars
    #request_file_from_filehandler_signal = pyqtSignal(int)  # Signal for image retrival

    def __init__(self, file_handling_thread):
        super().__init__()
        self.file_handling_thread = file_handling_thread

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

    def visualize_patches(self, image):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        window_size = 128 * 4
        min_padding_size = 8 * 4                     # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size

        #input_image_height = input_image_height * 4
        #input_image_width = input_image_width * 4


        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

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

        return image

    def update_preview_tile(self, patch_image, patch_number, file_list_item_number):
        self.preview_image = self.file_handling_thread.processing_file_list[file_list_item_number]["preview_file"]                    # Set the preview image to the image in the file handling thread
        input_image_width, input_image_height  = self.file_handling_thread.processing_file_list[file_list_item_number]["input_res"]                  # Get the input image resoloution from the file handling thread
        
        self.preview_image = self.visualize_patches(self.preview_image)
        input_image_height = input_image_height * 4
        input_image_width = input_image_width * 4
        
        window_size = 128   * 4                                      # Size of the window to be extracted from the image in pixels
        min_padding_size = 8   * 4                                   # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size               # Size of the patches to be extracted from the image in pixels

        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

        r = patch_number % number_of_windows_in_row
        c = patch_number // number_of_windows_in_row

        if r == number_of_windows_in_row - 1:
            x_start_point = (r * window_size) - (r * x_last_overlap)
        else:
            x_start_point = (r * window_size) - (r * x_overlap)

        if c == number_of_windows_in_col - 1:
            y_start_point = (c * window_size) - (c * y_last_overlap)
        else:
            y_start_point = (c * window_size) - (c * y_overlap)

        # add the patch image to the preview image in the correct location
        self.preview_image.paste(patch_image, (int(x_start_point), int(y_start_point)))

        # Draw a border around the patchwith the colour genrated from the patch number
        colour = plt.cm.jet(patch_number / (number_of_windows_in_col * number_of_windows_in_row))

        # Convert the float values to integers for the color tuple
        colour_int = tuple(int(x * 255) for x in colour[:-1])
        draw = ImageDraw.Draw(self.preview_image)
        draw.rectangle([x_start_point, y_start_point, x_start_point + window_size, y_start_point + window_size], width=10, outline=colour_int)

        # update the preview image in the file handling thread
        #self.file_handling_thread.processing_file_list[file_list_item_number]["preview_file"] = self.preview_image

        # send signal to update the preview image
        self.preview_update_signal.emit(self.preview_image)


#%%  - Load the UI file
Form, Window = uic.loadUiType("GUI\SDx4_interface.ui")
app = QApplication([])

from GUI.clickablewidget import ClickableWidget

class ThemeDesigner(QDialog):
    update_ui_preview_signal = pyqtSignal(dict)
    add_new_theme_signal = pyqtSignal(str)

    def __init__(self, current_ui_theme, current_ui_mode, avalible_themes, parent=None):
        super().__init__(parent)

        ThemeDesignerForm, ThemeDesignerWindow = uic.loadUiType(resource_path(r'App_Data\IntegratedThemeDesignerInterface.ui'))

        self.original_ui_theme = current_ui_theme
        self.original_ui_mode = current_ui_mode

        self.current_ui_theme = current_ui_theme
        self.current_ui_mode = current_ui_mode

        self.avalible_themes = avalible_themes

        self.ui = ThemeDesignerForm()

        self.ui.setupUi(self)
        self.setWindowTitle("Theme Designer")
        self.init_css_theme()
        self.init_signals()
        self.load_theme(self.current_ui_theme)
        
        # add the themes to the themes list selector
        self.ui.basethemesListSelector.addItems(self.avalible_themes)

        # Create list of false locks for each color group
        self.light_mode_locks = [False] * len(self.lighttheme_groups)
        self.dark_mode_locks = [False] * len(self.darktheme_groups)

    def init_css_theme(self):
        self.color_pick_buttons = {}
        self.color_pick_buttons["lightMainWidget"] = self.ui.lightMainWidget
        self.color_pick_buttons["lightSecondaryWidget"] = self.ui.lightSecondaryWidget
        self.color_pick_buttons["lightAccent1Widget"] = self.ui.lightAccent1Widget
        self.color_pick_buttons["lightAccent2Widget"] = self.ui.lightAccent2Widget
        self.color_pick_buttons["lightAccent3Widget"] = self.ui.lightAccent3Widget
        self.color_pick_buttons["lightAccent4Widget"] = self.ui.lightAccent4Widget
        self.color_pick_buttons["lightAccent5Widget"] = self.ui.lightAccent5Widget
        self.color_pick_buttons["lightText1Widget"] = self.ui.lightText1Widget
        self.color_pick_buttons["lightText2Widget"] = self.ui.lightText2Widget
        self.color_pick_buttons["lightIconsWidget"] = self.ui.lightIconsWidget

        self.color_pick_buttons["darkMainWidget"] = self.ui.darkMainWidget
        self.color_pick_buttons["darkSecondaryWidget"] = self.ui.darkSecondaryWidget
        self.color_pick_buttons["darkAccent1Widget"] = self.ui.darkAccent1Widget
        self.color_pick_buttons["darkAccent2Widget"] = self.ui.darkAccent2Widget
        self.color_pick_buttons["darkAccent3Widget"] = self.ui.darkAccent3Widget
        self.color_pick_buttons["darkAccent4Widget"] = self.ui.darkAccent4Widget
        self.color_pick_buttons["darkAccent5Widget"] = self.ui.darkAccent5Widget
        self.color_pick_buttons["darkText1Widget"] = self.ui.darkText1Widget
        self.color_pick_buttons["darkText2Widget"] = self.ui.darkText2Widget
        self.color_pick_buttons["darkIconsWidget"] = self.ui.darkIconsWidget

        self.color_pick_labels = {} 
        self.color_pick_labels["lightMainWidget"] = self.ui.lightMainLabel
        self.color_pick_labels["lightSecondaryWidget"] = self.ui.lightSecondaryLabel
        self.color_pick_labels["lightAccent1Widget"] = self.ui.lightAccent1Label
        self.color_pick_labels["lightAccent2Widget"] = self.ui.lightAccent2Label
        self.color_pick_labels["lightAccent3Widget"] = self.ui.lightAccent3Label
        self.color_pick_labels["lightAccent4Widget"] = self.ui.lightAccent4Label
        self.color_pick_labels["lightAccent5Widget"] = self.ui.lightAccent5Label
        self.color_pick_labels["lightText1Widget"] = self.ui.lightText1Label
        self.color_pick_labels["lightText2Widget"] = self.ui.lightText2Label
        self.color_pick_labels["lightIconsWidget"] = self.ui.lightIconsLabel

        self.color_pick_labels["darkMainWidget"] = self.ui.darkMainLabel
        self.color_pick_labels["darkSecondaryWidget"] = self.ui.darkSecondaryLabel
        self.color_pick_labels["darkAccent1Widget"] = self.ui.darkAccent1Label
        self.color_pick_labels["darkAccent2Widget"] = self.ui.darkAccent2Label
        self.color_pick_labels["darkAccent3Widget"] = self.ui.darkAccent3Label
        self.color_pick_labels["darkAccent4Widget"] = self.ui.darkAccent4Label
        self.color_pick_labels["darkAccent5Widget"] = self.ui.darkAccent5Label
        self.color_pick_labels["darkText1Widget"] = self.ui.darkText1Label
        self.color_pick_labels["darkText2Widget"] = self.ui.darkText2Label
        self.color_pick_labels["darkIconsWidget"] = self.ui.darkIconsLabel


        self.lighttheme_groups = ["lightMainWidget",
                                "lightSecondaryWidget",
                                "lightAccent1Widget",
                                "lightAccent2Widget",
                                "lightAccent3Widget",
                                "lightAccent4Widget",
                                "lightAccent5Widget",
                                "lightText1Widget",
                                "lightText2Widget",
                                "lightIconsWidget"]

        self.darktheme_groups = ["darkMainWidget",
                                "darkSecondaryWidget",
                                "darkAccent1Widget",
                                "darkAccent2Widget",
                                "darkAccent3Widget",
                                "darkAccent4Widget",
                                "darkAccent5Widget",
                                "darkText1Widget",
                                "darkText2Widget",
                                "darkIconsWidget"]

    def init_color_picker_button_colors(self):
        # load all the values from the theme color dictionaries into the lists
        self.dark_mode_colors = list(self.dark_theme_dictionary.values())
        self.light_mode_colors = list(self.light_theme_dictionary.values())

        # make a new list that is light theme colors + dark theme colors
        all_colors = self.light_mode_colors + self.dark_mode_colors 

        for color_pick_button, color_pick_label, color in zip(self.color_pick_buttons.values(), self.color_pick_labels.values(), all_colors):
            color_pick_button.setStyleSheet(f'background-color: {color};')
            color_pick_label.setText(color)

    def init_signals(self):
        self.ui.saveThemeBtn.clicked.connect(self.save_theme)
        self.ui.toggleEditModeBtn.clicked.connect(self.toggle_ui_mode)
        self.ui.randomAllUnlockedColorsBtn.clicked.connect(self.randomise_all_unlocked_colors)
        self.ui.basethemesListSelector.currentTextChanged.connect(self.select_base_theme)
        self.ui.colorMatchMethodListSelector.currentTextChanged.connect(self.set_color_match_method)

        self.ui.lightMainClickWidget.clicked.connect(self.color_picker)
        self.ui.lightSecondaryClickWidget.clicked.connect(self.color_picker)
        self.ui.lightAccent1ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightAccent2ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightAccent3ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightAccent4ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightAccent5ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightText1ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightText2ClickWidget.clicked.connect(self.color_picker)
        self.ui.lightIconsClickWidget.clicked.connect(self.color_picker)

        self.ui.darkMainClickWidget.clicked.connect(self.color_picker)
        self.ui.darkSecondaryClickWidget.clicked.connect(self.color_picker)
        self.ui.darkAccent1ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkAccent2ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkAccent3ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkAccent4ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkAccent5ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkText1ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkText2ClickWidget.clicked.connect(self.color_picker)
        self.ui.darkIconsClickWidget.clicked.connect(self.color_picker)

        self.ui.lightMainLockBtn.clicked.connect(self.lock_color)
        self.ui.lightSecondaryLockBtn.clicked.connect(self.lock_color)
        self.ui.lightAccent1LockBtn.clicked.connect(self.lock_color)
        self.ui.lightAccent2LockBtn.clicked.connect(self.lock_color)
        self.ui.lightAccent3LockBtn.clicked.connect(self.lock_color)
        self.ui.lightAccent4LockBtn.clicked.connect(self.lock_color)
        self.ui.lightAccent5LockBtn.clicked.connect(self.lock_color)
        self.ui.lightText1LockBtn.clicked.connect(self.lock_color)
        self.ui.lightText2LockBtn.clicked.connect(self.lock_color)
        self.ui.lightIconsLockBtn.clicked.connect(self.lock_color)

        self.ui.darkMainLockBtn.clicked.connect(self.lock_color)
        self.ui.darkSecondaryLockBtn.clicked.connect(self.lock_color)
        self.ui.darkAccent1LockBtn.clicked.connect(self.lock_color)
        self.ui.darkAccent2LockBtn.clicked.connect(self.lock_color)
        self.ui.darkAccent3LockBtn.clicked.connect(self.lock_color)
        self.ui.darkAccent4LockBtn.clicked.connect(self.lock_color)
        self.ui.darkAccent5LockBtn.clicked.connect(self.lock_color)
        self.ui.darkText1LockBtn.clicked.connect(self.lock_color)
        self.ui.darkText2LockBtn.clicked.connect(self.lock_color)
        self.ui.darkIconsLockBtn.clicked.connect(self.lock_color)

        self.ui.lightMainRandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightSecondaryRandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightAccent1RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightAccent2RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightAccent3RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightAccent4RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightAccent5RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightText1RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightText2RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.lightIconsRandomBtn.clicked.connect(self.randomise_single_color)

        self.ui.darkMainRandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkSecondaryRandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkAccent1RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkAccent2RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkAccent3RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkAccent4RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkAccent5RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkText1RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkText2RandomBtn.clicked.connect(self.randomise_single_color)
        self.ui.darkIconsRandomBtn.clicked.connect(self.randomise_single_color)

    def select_base_theme(self):
        theme = self.ui.basethemesListSelector.currentText()   

        # load the theme css into the strings
        self.load_theme(theme)

    def load_theme(self, theme):
        self.current_ui_theme = theme

        # load the css files into the strings
        with open(resource_path(f'App_Data/themes/{theme}/{theme}_dark_theme_dictionary.json'), 'r') as file:
            self.dark_theme_dictionary = json.load(file)

        with open(resource_path(f'App_Data/themes/{theme}/{theme}_light_theme_dictionary.json'), 'r') as file:
            self.light_theme_dictionary = json.load(file)

        # load all the values from the theme color dictionaries into the lists
        self.dark_mode_colors = list(self.dark_theme_dictionary.values())
        self.light_mode_colors = list(self.light_theme_dictionary.values())

        # update the ui preview with the new theme
        self.init_color_picker_button_colors()

        if self.current_ui_mode == "dark":
            self.update_ui_preview_signal.emit(self.dark_theme_dictionary)
        else:
            self.update_ui_preview_signal.emit(self.light_theme_dictionary)

    def lock_color(self):
        original_sender_name = self.sender().objectName()
        sender_name = original_sender_name
        sender_name = sender_name.replace("LockBtn", "")

        # Check sender name for light or dark and remove it and flag which one was found
        if "light" in sender_name:
            sender_name = sender_name.replace("light", "")
            for i, key in enumerate(self.lighttheme_groups):
                if sender_name.lower() in key.lower():
                    break

            # check if original sender Qpushbutton is checked or unchecked
            if self.sender().isChecked():
                self.light_mode_locks[i] = True
            else:
                self.light_mode_locks[i] = False

        else:
            sender_name = sender_name.replace("dark", "")
            for i , key in enumerate(self.darktheme_groups):
                if sender_name.lower() in key.lower():
                    break
            if self.sender().isChecked():
                self.dark_mode_locks[i] = True
            else:
                self.dark_mode_locks[i] = False

    def generate_random_color(self):
        # generate a random color hex code
        color = '#{:02x}{:02x}{:02x}'.format(*np.random.choice(range(256), size=3))
        return color
    
    def randomise_single_color(self):
        original_sender_name = self.sender().objectName()
        sender_name = original_sender_name
        sender_name = sender_name.replace("RandomBtn", "")

        # turn the keys of the dictionaries into lists
        self.light_theme_dictionary_keys = list(self.light_theme_dictionary.keys())
        self.dark_theme_dictionary_keys = list(self.dark_theme_dictionary.keys())

        # Check sender name for light or dark and remove it and flag which one was found
        if "light" in sender_name:
            sender_name = sender_name.replace("light", "")
            for lock, key in zip(self.light_mode_locks, self.light_theme_dictionary_keys):
                if sender_name.lower() in key.lower():
                    if lock == False:
                        print("Unlocked")
                        self.light_theme_dictionary[key] = self.generate_random_color()
                        break

        else:
            sender_name = sender_name.replace("dark", "")
            for lock, key in zip(self.dark_mode_locks, self.dark_theme_dictionary_keys):
                if sender_name.lower() in key.lower():
                    if lock == False:
                        print("Unlocked")
                        self.dark_theme_dictionary[key] = self.generate_random_color()
                        break

        self.init_color_picker_button_colors()

    def randomise_all_unlocked_colors(self):
        
        # turn the keys of the dictionaries into lists
        self.light_theme_dictionary_keys = list(self.light_theme_dictionary.keys())
        self.dark_theme_dictionary_keys = list(self.dark_theme_dictionary.keys())


        for lock, key in zip(self.light_mode_locks, self.light_theme_dictionary_keys):
            if lock == False:
                print("Unlocked")
                self.light_theme_dictionary[key] = self.generate_random_color()


        for lock, key in zip(self.dark_mode_locks, self.dark_theme_dictionary_keys):
            if lock == False:
                print("Unlocked")
                self.dark_theme_dictionary[key] = self.generate_random_color()

        self.init_color_picker_button_colors()

        if self.current_ui_mode == "dark":
            self.update_ui_preview_signal.emit(self.dark_theme_dictionary)
        
        else:
            self.update_ui_preview_signal.emit(self.light_theme_dictionary)

    def color_picker(self):
        color_dialog = QColorDialog(self)
        color_dialog.setWindowTitle('Choose Color')
        original_sender_name = self.sender().objectName()

        sender_name = original_sender_name
        sender_name = sender_name.replace("Click", "")
        sender_name = sender_name.replace("Widget", "")

        # Check sender name for light or dark and remove it and flag which one was found
        if "light" in sender_name:
            update = "light"
            sender_name = sender_name.replace("light", "")
            for key, value in self.light_theme_dictionary.items():
                if sender_name.lower() in key.lower():
                    break
            
            initial_color = QColor(self.light_theme_dictionary[key])

        else:
            update = "dark"
            sender_name = sender_name.replace("dark", "")
            # find the key in the dictionary that contains the entire sender name, check should be performed entirely in lower case
            for key, value in self.dark_theme_dictionary.items():
                if sender_name.lower() in key.lower():
                    break
            
            initial_color = QColor(self.dark_theme_dictionary[key])
        

        if initial_color.isValid():
            color_dialog.setCurrentColor(initial_color)

        if color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            color = color_dialog.selectedColor()

        else:
            return
        
        if update == "light":
            self.light_theme_dictionary[key] = color.name()
            self.update_ui_preview_signal.emit(self.light_theme_dictionary)
        else:
            self.dark_theme_dictionary[key] = color.name()
            self.update_ui_preview_signal.emit(self.dark_theme_dictionary)

        self.init_color_picker_button_colors()

        # update the color of the button
        #original_sender_name.setStyleSheet(f'background-color: {color.name()};')

        # Update the hex readout of the button to the color set
        #original_sender_name.setText(color.name())

    def toggle_ui_mode(self):
        if self.current_ui_mode == "dark":
            self.current_ui_mode = "light"
            #self.ui_preview.setStyleSheet(self.light_mode_css_content)
            # send ui mode change signal to main window
            self.update_ui_preview_signal.emit(self.light_theme_dictionary)

        else:
            self.current_ui_mode = "dark"
            #self.ui_preview.setStyleSheet(self.dark_mode_css_content)
            # send ui mode change signal to main window
            self.update_ui_preview_signal.emit(self.dark_theme_dictionary)

    def save_theme(self):
        self.ui.themeNameInput.setPlaceholderText("")
        
        # check if text has been enteres into the theme name box
        if self.ui.themeNameInput.text() == "":
            self.ui.themeNameInput.setPlaceholderText("A name is required to save!")
            return
        
        # get the theme name from the text box
        theme_name = self.ui.themeNameInput.text()

        # create a new folder in the themes folder with the name of the theme if the folder doesn't already exist
        os.makedirs(resource_path(f'App_Data/themes/{theme_name}'), exist_ok=True)

        # save the dictionary to a file so it can be loaded later as a dictionary easily
        with open(resource_path(f'App_Data/themes/{theme_name}/{theme_name}_dark_theme_dictionary.json'), 'w') as f:
            json.dump(self.dark_theme_dictionary, f)

        # save the dictionary to a file so it can be loaded later as a dictionary easily
        with open(resource_path(f'App_Data/themes/{theme_name}/{theme_name}_light_theme_dictionary.json'), 'w') as f:
            json.dump(self.light_theme_dictionary, f)

        # add the theme to the avalible themes list
        #self.avalible_themes.append(theme_name)
        self.ui.basethemesListSelector.addItems([theme_name])
        self.add_new_theme_signal.emit(theme_name)

        self.original_ui_theme = theme_name # update the original theme name to the new theme name

    def set_color_match_method(self):
        self.color_match_method = self.ui.colorMatchMethodListSelector.currentText()
        #FINISH!!! 

class ExitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        exitDialogForm, exitDialogWindow = uic.loadUiType(resource_path(r'App_Data\exitDialogInterface.ui'))
        self.ui = exitDialogForm()
        self.ui.setupUi(self)
        self.setWindowTitle("Exit Confirmation")

    def accept(self):
        self.accepted = True
        super().accept()
        sys.exit()

    def reject(self):
        self.accepted = False
        super().reject()

class CancelUpscaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        cancelUpscaleDialogForm, cancelUpscaleDialogWindow = uic.loadUiType(resource_path(r'App_Data\cancelUpscaleDialogInterface.ui'))
        self.ui = cancelUpscaleDialogForm()
        self.ui.setupUi(self)
        self.setWindowTitle("Cancel Upscale Confirmation")

    def accept(self):
        self.accepted = True
        super().accept()

    def reject(self):
        self.accepted = False
        super().reject()


        
## MAIN WINDOW CLASS
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Form()
        self.ui.setupUi(self)

        # Simple Customisation Settings
        self.online_version_file_link = "https://github.com/Adillwma/BackupInspector/raw/main/config.json"
        self.update_download_link = "https://github.com/Adillwma/BackupInspector/raw/main/BackupInspector.exe"

        self.config_file = resource_path(r'App_Data\config.json')
        self.help_text_path = resource_path(r"App_Data\copy\help_text.txt")
        self.info_text_path = resource_path(r"App_Data\copy\info_text.txt")
        self.temp_folder = resource_path(r"App_Data\temp_data")   
        self.main_css_path = resource_path(r"App_Data\mainStylesheet.css")

        ### Initialise UI Elements
        self.check_preferences()          # Checks the user preferences file to load the correct settings
        self.init_icons()                 # Load the icons for the buttons
        self.init_modularUI()           # Modular Unified Ui
        self.init_programUI()           # Main Program Ui elements
        self.initalise_settings()          # User Settings 
        self.initialise_avalible_theme_list()
        
        # Begin program threads
        self.file_handling_thread = FileHandlingThread(self.temp_folder)
        self.is_upscale_running = False

        # Initialise Backend
        self.file_processing_thread = FileProcessingThread()

        # Connect signals from the file processing thread to gui
        self.file_processing_thread.finished.connect(self.file_processing_finished)
        self.file_processing_thread.stopped.connect(self.file_processing_stopped)
        self.file_processing_thread.processing_position_signal.connect(self.update_progress_bars)

        # Initialise the upscale preview thread
        self.upscale_preview_thread = UpscalePreviewThread(self.file_handling_thread)
        self.upscale_preview_thread.preview_update_signal.connect(self.refresh_plot_data)
        
        self.file_processing_thread.patch_preview.connect(self.upscale_preview_thread.update_preview_tile)
        self.file_processing_thread.tile_complete_signal.connect(self.upscale_preview_thread.update_preview_tile)

        #self.file_handling_thread.start()
        self.upscale_preview_thread.start()



    #%% - Initialise UI
    def check_preferences(self):                                                            # Checks the user preferences file to load the correct settings
        # Load data from the config file
        with open(self.config_file, 'r') as file:
            self.config_data = json.load(file)

        # check if the program has been run before if not run wizard for user to set preferences
        if self.config_data["First Run"] == True:
            self.config_data = self.show_setup_wizard()                          # show the setup wizard

        # Program Settings
        self.version_number = self.config_data["Version"]                        # set the version number for checking for updates

        # UI Visual Settings
        self.current_theme = self.config_data["Theme"]                     # set the theme to user preference
        self.ui.themesListSelector.setCurrentText(self.current_theme) 
        #self.current_ui_mode = self.config_data["UI"]                     # set the ui to user preference dark/light mode
        #self.                 

        # Directory Paths
        self.output_dir = self.config_data["output_dir"]   # set the output folder path to user preference
        
        # Main Processor States
        self.blending = self.config_data["blending"]         # set the verify checksum state to user preference
        self.blend_mode = self.config_data["blend_mode"]             # set the checksum type to user preference
        self.num_inference_steps = self.config_data["num_inference_steps"]       # set the target bit depth to user preference
        self.guidance_scale = self.config_data["guidance_scale"]       # set the target bit depth to user preference
        self.boost_face_quality = self.config_data["boost_face_quality"]       # set the target bit depth to user preference

        # Pipeline Settings
        self.cpu_offload = self.config_data["cpu_offload"]     # set the normalise up only state to user preference
        self.attention_slicing = self.config_data["attention_slicing"]   # set the target sample rate to user preference
        self.xformers = self.config_data["xformers"]       # set the target bit depth to user preference

    def init_modularUI(self):
        # Left Menu
        self.left_menu_animation = QPropertyAnimation(self.ui.leftMenuContainer, b"maximumWidth")
        self.left_menu_animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self.left_menu_animation.setDuration(1000)  # Animation duration in milliseconds
        self.ui.leftMenuBtn_UiBtnType.clicked.connect(self.expandorshrink_left_menu)
        self.ui.leftMenuContainer.setMaximumWidth(50)         

        self.ui.settingsBtn_UiBtnType.clicked.connect(lambda: self.handle_centre_menu(page=self.ui.settingsCenterMenuPage))
        self.ui.infoBtn_UiBtnType.clicked.connect(lambda: self.handle_centre_menu(page=self.ui.infoCenterMenuPage))
        self.ui.helpBtn_UiBtnType.clicked.connect(lambda: self.handle_centre_menu(page=self.ui.helpCenterMenuPage))

        # Center Menu
        self.center_menu_animation = QPropertyAnimation(self.ui.centerMenuContainer, b"maximumWidth")
        self.center_menu_animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self.center_menu_animation.setDuration(1000)  # Animation duration in milliseconds
        self.ui.centerMenuCloseBtn_UiBtnType.clicked.connect(lambda: self.run_animation(self.center_menu_animation, start=250, end=5))
        self.ui.centerMenuContainer.setMaximumWidth(5)         # Set centre menu to start hidden (with max width of 0)   

        # Notification Container
        self.notification_animation = QPropertyAnimation(self.ui.popupNotificationContainer, b"maximumHeight")
        self.notification_animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self.notification_animation.setDuration(1000)
        self.ui.notificationCloseBtn_UiBtnType.clicked.connect(lambda: self.run_animation(self.notification_animation, start=100, end=0))
        self.ui.popupNotificationContainer.setMaximumHeight(0)  # Set notification container to start hidden (with max height of 0)

        # ui theme dark / light
        self.dark_mode_path = resource_path(fr"App_Data\themes\{self.current_theme}\dark_mode.css")
        self.light_mode_path = resource_path(fr"App_Data\themes\{self.current_theme}\light_mode.css")
        self.current_ui_mode = "dark"
        self.highlight_theme_color = "background-color: #1f232a;"
        self.ui.uiThemeBtn_UiBtnType.clicked.connect(self.switch_ui_mode)
        self.set_icons_color("#FFFFFF")   #CHANGE COLOUR TO COME FROM STYLE SHEET PROGRAMATICALLY!!!!
        self.set_theme()

        ### SETTINGS PAGE
        self.ui.themesListSelector.currentTextChanged.connect(self.set_theme)
        self.ui.checkForUpdates_ProgramBtnType.clicked.connect(self.check_online_for_updates)
        self.ui.downloadUpdateBtn_ProgramBtnType.clicked.connect(self.download_latest_version)
        self.ui.runThemeDesigner_ProgramBtnType.clicked.connect(self.open_theme_designer_dialog)

        ### HELP & INFO PAGES
        self.set_helpandinfo_copy(self.help_text_path, self.info_text_path)

    def init_programUI(self):
        self.upscale_settings_animation = QPropertyAnimation(self.ui.upscaleSettingsWidget, b"maximumWidth")
        self.upscale_settings_animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self.upscale_settings_animation.setDuration(1000)  # Animation duration in milliseconds
        self.ui.advancedmodeBtn.clicked.connect(self.toggle_advanced_mode)
        #self.ui.upscaleSettingsWidget.setMaximumWidth(0)         # Set upscale settings widget to start hidden (with max width of 0)

        self.ui.iterationsSlider.valueChanged.connect(self.update_iterations_setting)
        self.ui.guidanceSlider.valueChanged.connect(self.update_guidance_settings)

        self.ui.blendingCheckbox.clicked.connect(self.toggle_blending)
        self.ui.blendTypeSelector.currentTextChanged.connect(self.toggle_blending_mode)
        self.ui.boostfaceQualityCheckbox.clicked.connect(self.toggle_boostface_quality)
        self.ui.cpuoffloadCheckbox.clicked.connect(self.toggle_cpu_offload)
        self.ui.attentionSlicingCheckbox.clicked.connect(self.toggle_attentionslicing)
        self.ui.xformersCheckbox.clicked.connect(self.toggle_xformers)

        self.ui.outputLocationBrowseBtn.clicked.connect(self.browse_output_location)
        self.ui.addFilesBtn.clicked.connect(self.browse_input_files)
        self.ui.addfoldersBtn.clicked.connect(self.browse_input_folders)
        self.ui.removeListItemBtn.clicked.connect(self.remove_selected_list_item)
        self.ui.runUpscaleBtn.clicked.connect(self.upscale_btn_clicked)

        # if a user has selcted an item in the list connect the itemClicked signal to the display image function
        self.ui.inputFilesListDisplay.itemClicked.connect(self.file_selected_in_list)

        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout for the widget_28 if it's not a layout already
        layout = QVBoxLayout(self.ui.widget_28)

        # Add the canvas to the layout
        layout.addWidget(self.canvas)
        self.user_text_warning = self.ax.text(0.5, 0.5, 'Please select an image', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=20)              # add text on the plot to tell the user to select an image
        self.ax.axis('off')                                 # turn off axis
        self.figure.patch.set_alpha(0)                      # set mpl background alpha to 0 to make it transparent 
        self.canvas.draw()                 # Refresh the canvas 

        # create a label to display the image
        self.image_label = ImageWidget(self)
        layout.addWidget(self.image_label)
        self.image_label.hide()           # hide the image label element for now until an image is selected


    def init_icons(self):
        self.buttons = [self.ui.leftMenuBtn_UiBtnType,
                        self.ui.settingsBtn_UiBtnType,
                        self.ui.infoBtn_UiBtnType,
                        self.ui.helpBtn_UiBtnType,
                        self.ui.centerMenuCloseBtn_UiBtnType,
                        self.ui.notificationCloseBtn_UiBtnType,
                        self.ui.advancedmodeBtn,
                        self.ui.addFilesBtn,
                        self.ui.addfoldersBtn, 
                        self.ui.removeListItemBtn,
                        self.ui.runUpscaleBtn, 

                        self.ui.makeSettingsDefaultCheckbox,
                        self.ui.cpuoffloadCheckbox,
                        self.ui.attentionSlicingCheckbox,
                        self.ui.xformersCheckbox,
                        self.ui.blendingCheckbox,
                        self.ui.boostfaceQualityCheckbox,
                        self.ui.uiThemeBtn_UiBtnType,
                        ]
        
        # find way to do this from the resources file ??? if not them move the required icons to icon folder in app data and load from there using the resource path fucntion
        self.svg_files = [[r"App_Data\icons\light\align-justify.svg"],
                          [r"App_Data\icons\light\settings.svg"],
                          [r"App_Data\icons\light\info.svg"],
                          [r"App_Data\icons\light\help-circle.svg"],
                          [r"App_Data\icons\light\x-circle.svg"],
                          [r"App_Data\icons\light\x-circle.svg"],   
                          [r"App_Data\icons\light\star.svg"],
                          [r"App_Data\icons\light\check-square.svg"],
                          [r"App_Data\icons\light\clipboard.svg"],
                          [r"App_Data\icons\light\trash-2.svg"],
                          [r"App_Data\icons\light\maximize-2.svg"],

                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],
                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],
                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],
                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],
                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],
                          [r"App_Data\icons\light\toggle-left.svg", r"App_Data\icons\light\toggle-right.svg"],

                          [r"App_Data\icons\light\sun.svg", r"App_Data\icons\light\moon.svg"],
                        ]

    def initalise_settings(self): 
        # User Selectable
        self.ui.blendingCheckbox.setChecked(self.blending)
        self.ui.blendTypeSelector.setCurrentText(self.blend_mode)
        self.ui.cpuoffloadCheckbox.setChecked(self.cpu_offload)
        self.ui.attentionSlicingCheckbox.setChecked(self.attention_slicing)
        self.ui.xformersCheckbox.setChecked(self.xformers)
        self.ui.boostfaceQualityCheckbox.setChecked(self.boost_face_quality)    # Create in GUI and connect!!!!

        self.ui.outputLocationTextDisplay.setText(self.output_dir)
        self.ui.imageProgressBar.setValue(0)
        self.ui.tileProgressBar.setValue(0)
        self.ui.iterationProgressBar.setValue(0)

        self.ui.iterationsSlider.setValue(self.num_inference_steps)
        self.ui.guidanceSlider.setValue(self.guidance_scale)

        self.ui.iterationsLabel.setText(f"Iterations: {self.num_inference_steps}")
        self.ui.guidanceLabel.setText(f"Guidance: {self.guidance_scale}")

        # Internal Settings
        self.callback_steps = 1
        self.patch_size = 120
        self.padding_size = 8
        self.safety_checker = None
        self.seed = None
        self.prompt = ""
        self.negative_prompt = ""
        self.show_patches = False
        self.dummy_upscale = False
        #self.boost_face_quality = True   # THSI CURRENTLY OVERIDES THE LOADED SETTING BUT IM USING FOR DEBUGGING!! , REMOVE WHEN DONE

        self.displayed_file = None

    def initialise_avalible_theme_list(self):
        # load all the avalible themes from the themes folder
        self.avalible_themes = os.listdir(resource_path(r"App_Data\themes"))

        # Remove the 'default' theme from the list as it aleady exists in the program
        self.avalible_themes.remove("default")

        # add the themes to the themes list selector
        self.ui.themesListSelector.addItems(self.avalible_themes)

    def set_helpandinfo_copy(self, help_file_path, info_file_path):
        with open(help_file_path, "r") as file:
            self.ui.helpTextCopy.setText(self.ui.helpTextCopy.toHtml().replace("Help file failed to load. Sorry for the mix-up, we are scrambling to fix it!", file.read()))

        with open(info_file_path, "r") as file:
            self.ui.infoTextCopy.setText(self.ui.infoTextCopy.toHtml().replace("Information file failed to load. Sorry for the mix-up, we are scrambling to fix it!", file.read()))

    #%% - Ui Functions
    def run_animation(self, animation_item, start, end):
        animation_item.setStartValue(start)
        animation_item.setEndValue(end)
        animation_item.start()

    def expandorshrink_left_menu(self):
        # If the left menu is closed (width = 40), then it is opened
        if self.ui.leftMenuContainer.maximumWidth() == 50:
            self.run_animation(self.left_menu_animation, start=50, end=250)
        elif self.ui.leftMenuContainer.maximumWidth() == 250:
            self.run_animation(self.left_menu_animation, start=250, end=50)
        else:     
            pass    # Ignore presses whilst animation is running


    def handle_centre_menu(self, page):
        # if the button pressed is the same as the current page and the menu is open then close the center menu
        if self.ui.centerMenuPagesStack.currentWidget() == page and self.ui.centerMenuContainer.maximumWidth() != 5:
            self.run_animation(self.center_menu_animation, start=250, end=5)  
        else:
            # If the center menu is closed (width = 5), then it is opened
            if self.ui.centerMenuContainer.maximumWidth() == 5:
                self.run_animation(self.center_menu_animation, start=5, end=250)

            # The correct center menu page is set for centerMenuPagesStack
            self.ui.centerMenuPagesStack.setCurrentWidget(page)

    def check_online_for_updates(self):
        '''Checks online for updates and displays a notification if there is one'''

        # Get the latest version from the github repo json file
        response = requests.get(self.online_version_file_link)    # Send an HTTP GET request to the .JSON URL
        if response.status_code == 200:   # Verify request was successful (HTTP status code 200)
            text_content = response.text   # Read the response into a text string
        else:
            self.ui.popupNotificationText.setText(f"Unable to check for updates. HTTP status code: {response.status_code}")   # Report HTTP error and code
            self.run_animation(self.notification_animation, start=0, end=100)
            return
        
        latest_config_data = json.loads(text_content) # load as json data with 'loads' for string
        
        latest_version = latest_config_data["Version"]
        latest_version_int = int(''.join(latest_version.split('.')))

        current_version_int = int(''.join(self.version_number.split('.')))

        if current_version_int < latest_version_int:
            self.ui.popupNotificationText.setText(f"New version available: {latest_version}, please download update")
            self.run_animation(self.notification_animation, start=0, end=100)
            self.ui.downloadUpdateBtn_ProgramBtnType.setEnabled(True)   # Enable the update button

        else:
            self.ui.popupNotificationText.setText("You are using the latest version!")
            self.run_animation(self.notification_animation, start=0, end=100)

    def download_latest_version(self):
        # open link in users default browser
        os.startfile(self.update_download_link)

    def closeEvent(self, event):
        event.ignore()  # Ignore the close event
        self.exit_dialog = ExitDialog(self)
        self.exit_dialog.exec()

    def switch_ui_mode(self):
        if self.current_ui_mode == "light":
            self.current_ui_mode = "dark"

        elif self.current_ui_mode == "dark":
            self.current_ui_mode = "light"
            
        self.set_theme()

    def set_theme(self):
        '''Sets the theme of the UI based on the selected theme in the themesListSelector'''

        # Get the selected theme from the themesListSelector
        selected_theme = self.ui.themesListSelector.currentText()

        # Set the current theme to the selected theme
        self.current_theme = selected_theme

        # Load the data from the main.css file as a string
        with open(self.main_css_path, 'r') as f:
            raw_css_data = f.read()

        # Load the data from the json file as a dictionary
        with open(resource_path(f'App_Data\\themes\\{self.current_theme}\{self.current_theme}_{self.current_ui_mode}_theme_dictionary.json'), 'r') as f:
            theme_dictionary = json.load(f)

        # Parse the main css file and add in the theme colors
        processed_css = self.custom_qss_css_parser(theme_dictionary, raw_css_data)
        
        # Set the stylesheet of the main window to the processed css
        self.setStyleSheet(processed_css)
        
        # Set the svg icons to the correct color based on the theme
        self.set_icons_color(theme_dictionary["@iconsColor"])    # Black   (Change to read theme ccss file for the right dark thme icon colour)

        # Save the current theme as default startup theme in the config JSON file
        with open(self.config_file, 'r') as file:
            data = json.load(file)

        data["Theme"] = self.current_theme   # Update the "Theme" value

        with open(self.config_file, 'w') as file:
            json.dump(data, file)  

    def custom_qss_css_parser(self, theme_dictionary, css_data):
        '''Parses the main.css file and adds in the theme colors'''

        # Search the data string for the keys in the dictionary and replace them with the values
        for key in theme_dictionary:
            css_data = css_data.replace(key, theme_dictionary[key])

        return css_data

    def set_icons_color(self, color_hex):
        for svg_files, button in zip(self.svg_files, self.buttons):
            pixmaps = []

            for svg_file in svg_files:
                # Load the SVG file into a QDomDocument
                document = QDomDocument()
                with open(resource_path(svg_file), 'r') as file:
                    document.setContent(file.read())
                
                # Modify the SVG stroke colour in memory, leaving file on disk unchanged
                # Define a list of element types
                element_types = ["path", "circle", "rect", "line", "polyline", "polygon", "ellipse"]

                # Loop through element types in the svg files
                for element_type in element_types:
                    # Find elements of the current type
                    elements = document.elementsByTagName(element_type)
                    
                    # Loop through elements of the current type
                    for i in range(elements.length()):
                        current_element = elements.item(i).toElement()
                        current_element.setAttribute("stroke", color_hex)

                # Create a QPixmap from the modified SVG document
                renderer = QSvgRenderer(document.toByteArray())
                pixmap = QPixmap(renderer.defaultSize())

                # Set pixmap background transparent
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()

                # Add the QPixmap to the list of pixmaps
                pixmaps.append(pixmap)

            if len(pixmaps) == 1:
                button.setIcon(QIcon(pixmaps[0]))

            elif len(pixmaps) == 2:
                togglable_icon = QIcon()
                togglable_icon.addPixmap(pixmaps[0], QIcon.Mode.Normal, QIcon.State.Off)
                togglable_icon.addPixmap(pixmaps[1], QIcon.Mode.Normal, QIcon.State.On)
                button.setIcon(togglable_icon)

    def open_theme_designer_dialog(self):
        Dialog = ThemeDesigner(self.current_theme, self.current_ui_mode, self.avalible_themes)
        Dialog.update_ui_preview_signal.connect(self.preview_from_theme_creator)
        Dialog.add_new_theme_signal.connect(self.add_new_theme)
        Dialog.exec()
        
        # Set the theme to the original theme before theme creator was opened if no theme was saved or the last saved theme if usr used theme creator made thme and saved it
        self.ui.themesListSelector.setCurrentText(Dialog.original_ui_theme)
        self.set_theme()

    def preview_from_theme_creator(self, theme_dictionary):
        # Load the data from the main.css file as a string
        with open(self.main_css_path, 'r') as f:
            raw_css_data = f.read()

        # Parse the main css file and add in the theme colors
        processed_css = self.custom_qss_css_parser(theme_dictionary, raw_css_data)
        
        # Set the stylesheet of the main window to the processed css
        self.setStyleSheet(processed_css)
        
        # Set the svg icons to the correct color based on the theme
        self.set_icons_color(theme_dictionary["@iconsColor"])    # Black   (Change to read theme ccss file for the right dark thme icon colour)

    def add_new_theme(self, theme_name):
        self.ui.themesListSelector.addItems([theme_name])
        self.ui.themesListSelector.setCurrentText(theme_name)

    #%% - Main Program Functions      
    def toggle_advanced_mode(self):
        # If the left menu is closed (width = 40), then it is opened
        if self.ui.upscaleSettingsWidget.maximumWidth() == 0:
            self.run_animation(self.upscale_settings_animation, start=0, end=345)
            self.ui.advancedmodeBtn.setText("Switch to Easy Mode")
        else:
            self.run_animation(self.upscale_settings_animation, start=345, end=0)
            self.ui.advancedmodeBtn.setText("Switch to Advanced Mode")

    def toggle_blending(self):
        if self.ui.blendingCheckbox.isChecked():
            self.blending = True
            self.ui.blendingWidgetCard.setStyleSheet("background-color: #2e3440;")
            self.ui.blendTypeWidgetCard.setStyleSheet("background-color: #2e3440;")
        else:
            self.blending = False
            self.ui.blendingWidgetCard.setStyleSheet("background-color: #1f232a;")
            self.ui.blendTypeWidgetCard.setStyleSheet("background-color: #1f232a;")

    def toggle_blending_mode(self):
        self.blend_mode = self.ui.blendTypeSelector.currentText()

    def toggle_boostface_quality(self):
        if self.ui.boostfaceQualityCheckbox.isChecked():
            self.boost_face_quality = True
            self.ui.boostfaceWidgetCard.setStyleSheet("background-color: #2e3440;")
        else:
            self.boost_face_quality = False
            self.ui.boostfaceWidgetCard.setStyleSheet("background-color: #1f232a;")

    def toggle_cpu_offload(self):
        if self.ui.cpuoffloadCheckbox.isChecked():
            self.cpu_offload = True
            self.ui.cpuoffloadWidgetCard.setStyleSheet("background-color: #2e3440;")
        else:
            self.cpu_offload = False
            self.ui.cpuoffloadWidgetCard.setStyleSheet("background-color: #1f232a;")

    def toggle_attentionslicing(self):
        if self.ui.attentionSlicingCheckbox.isChecked():
            self.attention_slicing = True
            self.ui.attentionslicingWidgetCard.setStyleSheet("background-color: #2e3440;")
        else:
            self.attention_slicing = False
            self.ui.attentionslicingWidgetCard.setStyleSheet("background-color: #1f232a;")

    def toggle_xformers(self):
        if self.ui.xformersCheckbox.isChecked():
            self.xformers = True
            self.ui.xformersWidgetCard.setStyleSheet("background-color: #2e3440;")
        else:
            self.xformers = False
            self.ui.xformersWidgetCard.setStyleSheet("background-color: #1f232a;")

    def update_iterations_setting(self):
        self.num_inference_steps = self.ui.iterationsSlider.value()
        self.ui.iterationsLabel.setText(f"Iterations: {self.ui.iterationsSlider.value()}")

    def update_guidance_settings(self):
        self.guidance_scale = self.ui.guidanceSlider.value()
        self.ui.guidanceLabel.setText(f"Guidance: {self.ui.guidanceSlider.value()}")

    def browse_output_location(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        self.ui.outputLocationTextDisplay.setText(self.output_dir)

    def browse_input_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Input Files", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_paths:              # if user selects a path wothout canceling the dialog box
            self.ui.inputFilesListDisplay.addItems(file_paths)
            self.file_handling_thread.add_files(file_paths)

    def browse_input_folders(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directories")
        if folder_path:              # if user selects a path wothout canceling the dialog box
            self.ui.inputFilesListDisplay.addItems(os.listdir(folder_path))     # add the filenames to the display list 
            self.file_handling_thread.add_folder(folder_path)

    def remove_selected_list_item(self):
        # if a user has selcted an item in the list then remove it from the list and the processing file list and remove its images from the temp folder 
        if self.ui.inputFilesListDisplay.currentItem():
            self.ui.inputFilesListDisplay.takeItem(self.ui.inputFilesListDisplay.currentRow())
            self.file_handling_thread.remove_file(self.ui.inputFilesListDisplay.currentRow())
            self.displayed_file = None
            self.update_plot_data(None)

    def file_selected_in_list(self):
        # If a file is selected in the list, set self.diplayed_file to the full file path of the selected file
        if self.ui.inputFilesListDisplay.currentItem():
            self.displayed_file = self.ui.inputFilesListDisplay.currentRow()
            file_size, input_res, output_res = self.file_handling_thread.get_file_info(self.displayed_file)

            self.update_plot_data(self.displayed_file)    # if the displayed file is not None then display it in the image viewer

            self.ui.filesizeDisplayLabel.setText(f"{file_size}")
            self.ui.inputresDisplayLabel.setText(f"{input_res[0]} x {input_res[1]}")
            self.ui.outputresDisplayLabel.setText(f"{output_res[0]} x {output_res[1]}")


    ##RENAME AND REFACTOR FOLLOWING TWO FUNCTIONS TO SIMPLIFY 
    def update_plot_data(self, image_path):                # change name to set plot data
        if image_path is None:
            self.refresh_plot_data(None)  # Display an empty image or handle it based on your use case
            self.image_label.hide()
            self.canvas.show()
        else:
            _, preview_image = self.file_handling_thread.get_file_images(image_path)
            self.refresh_plot_data(preview_image)
            self.canvas.hide()
            self.image_label.show()

    def refresh_plot_data(self, image_input):             # change name to 
        if image_input is None:
            # Handle the case of no image (display an empty image or handle it based on your use case)
            image_pixmap = QPixmap()  
            return
        
        elif isinstance(image_input, Image.Image):  # Check if image_input is a PIL Image
            image_pixmap = QPixmap.fromImage(ImageQt(image_input))

        else:
            raise ValueError("Invalid input type")

        self.image_label.setPixmap(image_pixmap)


   
 
    def upscale_btn_clicked(self):
        if self.is_upscale_running == False:
            self.run_upscale()
            return

        # If the upscale is already running
        # launch a dialog box to confirm if the user wants to cancel the upscale
        self.cancel_upscale_dialog = CancelUpscaleDialog(self)
        self.cancel_upscale_dialog.exec()

        if self.cancel_upscale_dialog.accepted:
            self.cancel_upscale()
            print("Upscale cancelled")
        else:
            pass

    def cancel_upscale(self):
        # Request interruption to stop the file processing thread
        self.file_processing_thread.upscaler.interupt_requested = True

        # change the upscale button text to "Run Upscale"
        self.ui.runUpscaleBtn.setText("Run Upscale")

        # Update current upscale status
        self.is_upscale_running = False

    def run_upscale(self):
        # Update current upscale status
        self.is_upscale_running = True

        # disable all ui buttons from interaction other than the run upscale button


        # change the run upscale button text to "Stop Upscale"
        self.ui.runUpscaleBtn.setText("Cancel Upscale")

        if self.ui.makeSettingsDefaultCheckbox.isChecked():
            self.update_defaults()

        # Start the file processing thread
        self.start_file_processing_thread()

    def update_defaults(self):

        # Load data from the config file
        with open(self.config_file, 'r') as file:
            self.config_data = json.load(file)
        
        # Directory Paths
        self.config_data["output_dir"] = self.output_dir

        # Main Processor States
        self.config_data["blending"] = self.blending
        self.config_data["blend_mode"] = self.blend_mode
        self.config_data["num_inference_steps"] = self.num_inference_steps
        self.config_data["guidance_scale"] = self.guidance_scale

        # Pipeline Settings
        self.config_data["boost_face_quality"] = self.boost_face_quality
        self.config_data["cpu_offload"] = self.cpu_offload
        self.config_data["attention_slicing"] = self.attention_slicing
        self.config_data["xformers"] = self.xformers

        # Save the updated config data to the config file
        with open(self.config_file, 'w') as file:
            json.dump(self.config_data, file, indent=4)

    #%% - Backend Thread Functions
    def start_file_processing_thread(self):  
        
        self.file_processing_thread.initialize_upscale_job( self.file_handling_thread.processing_file_list,              # specify list of input images
                                                            self.output_dir,           # specify output directory
                                                            self.patch_size,         
                                                            self.padding_size,               
                                                            self.num_inference_steps,                # Must be > 3 for any usable output, > 5 for okay output, > 10 for good output, 50+ for great output
                                                            self.guidance_scale,          
                                                            self.prompt,                            # specify prompt for image upscaling
                                                            self.negative_prompt,                   # specify negative prompt for image upscaling
                                                            self.boost_face_quality,                                                                        
                                                            self.blending,                        # If True, will use soft blend. If False, will use hard blend. 
                                                            self.blend_mode,
                                                            self.callback_steps,                    # Number of steps between each callback. Set to 0 to disable callbacks. Callbacks are used to show the progress of the image upscaling.
                                                            self.show_patches,                   # If True, will show the patches that are being upscaled. If False, will not show the patches that are being upscaled.
                                                            self.dummy_upscale,                  # For debugging. If True, will not use the neural net to upscale image, instead a very very fast bicubic upscale is used, to speed up testing the rest of the code. Always set to False for actual usage.
                                                            self.xformers,                       # If True, will use the xformers model. If False, will use the original model.
                                                            self.cpu_offload,                    # If True, will use the CPU for the first few inference steps, then switch to the GPU. If False, will use the GPU for all inference steps. If True, will be slower but will use less GPU memory.
                                                            self.attention_slicing,              # If True, will use attention slicing. If False, will not use attention slicing. If True, will be slower but will use less GPU memory.
                                                            self.seed,                            # If None, will use a random seed. If set to a numerical value, will use value as the generator seed.
                                                            self.safety_checker)

        # Start the file processing thread run method
        self.file_processing_thread.start()



    def cancel_file_processing(self):
        # Request interruption to stop the file processing thread
        self.file_processing_thread.upscaler.interupt_requested = True

    def file_processing_finished(self):
        print("File processing finished")
        if self.ui.imageProgressBar.value() != 100:
            self.ui.imageProgressBar.setValue(100)

        if self.ui.tileProgressBar.value() != 100:
            self.ui.tileProgressBar.setValue(100)

        if self.ui.iterationProgressBar.value() != 100:
            self.ui.iterationProgressBar.setValue(100)

    def file_processing_stopped(self):
        print("File processing stopped")

    def update_progress_bars(self, image_num, total_images, tile_num, total_tiles, iteration_num, total_iterations):

        # Calulate the percentage of images, tiles and iterations processed
        image_percentage = int((image_num-1 / total_images) * 100)            
        tile_percentage = int((tile_num / total_tiles) * 100)            
        iteration_percentage = int(((iteration_num + 1) / total_iterations) * 100)          
        
        # Update the progress bars if the values have changed
        if self.ui.imageProgressBar.value() != image_percentage:
            self.ui.imageProgressBar.setValue(image_percentage)

        if self.ui.tileProgressBar.value() != tile_percentage:
            self.ui.tileProgressBar.setValue(tile_percentage)

        if self.ui.iterationProgressBar.value() != iteration_percentage:
            self.ui.iterationProgressBar.setValue(iteration_percentage)


#%% - Run Program
if __name__ == "__main__":
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



