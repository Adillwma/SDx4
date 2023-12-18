import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
from PyQt6 import uic
import resources_rc
import os
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
import math

import shutil

from customWidgets import CustomToggle



# trye to take the signals idretly form file_processing_thread.upscaler to the gui clss without going through the file_processing_thread classs, aslo then can xsperate the image complete proress signal freom the other two and put image complete after the image is porcessed in file_processing_thread

#%% Helper Functions
def resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#%% - Backend Program Functions

from tqdm import tqdm
from SDx4_Upscaler_Class import SDx4Upscaler

# File processing thread
class FileProcessingThread(QThread):
    processing_position_signal = pyqtSignal(int, int, int, int, int, int)  # Emits (current image, current tile, current iteration) during the upscaling process for gui progress bars
    patch_preview = pyqtSignal(object, int, str)  # Signal for previewing patches
    tile_complete_signal = pyqtSignal(object, int, str)  # Signal for tile complete image retrival
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, processing_file_list, output_dir, patch_size, padding_size, num_inference_steps, guidance_scale, prompt, negative_prompt, boost_face_quality, blending, blend_mode, callback_steps, show_patches, dummy_upscale, xformers, cpu_offload, attention_slicing, seed, safety_checker):
        super().__init__()
        self.local_image_paths = [file_info["input_file_path"] for file_info in processing_file_list]
        self.number_of_images = len(self.local_image_paths)
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.padding_size = padding_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.blending = blending
        self.blend_mode = blend_mode
        self.callback_steps = callback_steps
        self.show_patches = show_patches
        self.dummy_upscale = dummy_upscale
        self.xformers = xformers
        self.cpu_offload = cpu_offload
        self.attention_slicing = attention_slicing
        self.seed = seed
        self.boost_face_quality = boost_face_quality
        self.safety_checker = safety_checker
        self.upscale = SDx4Upscaler(self.xformers, self.cpu_offload, self.attention_slicing, self.seed, self.safety_checker)
        self.upscale.callback_signal.connect(self.send_patch_preview)
        self.upscale.tile_complete_signal.connect(self.send_tile_complete)
        self.upscale.processing_position_signal.connect(self.send_progress_update)
        os.makedirs(self.output_dir, exist_ok=True)

    def send_patch_preview(self, img_patch, patch_num, original_image):
        self.patch_preview.emit(img_patch, patch_num, original_image)
        #print("patch preview sent from processing thread")

    def send_tile_complete(self, tile_complete_image, tile_num, original_image):
        self.tile_complete_signal.emit(tile_complete_image, tile_num, original_image)
        #print("tile complete sent from processing thread")

    def send_progress_update(self, current_tile, total_tiles, current_iteration, total_iterations):
        self.processing_position_signal.emit(self.current_image, self.number_of_images, current_tile, total_tiles, current_iteration, total_iterations)
        #print("progress update sent from processing thread")

    def run(self):
        for current_image_num, local_image_path in enumerate(self.local_image_paths):
            # Check for interruption requests
            if self.isInterruptionRequested():    # need better method that can exit anytime during processing not just between files
                self.stopped.emit()
                return

            self.current_image = current_image_num + 1

            ### missing params i.e blend mode!!!
            upscaled_image = self.upscale.upscale(local_image_path, self.patch_size, self.padding_size, self.num_inference_steps, self.guidance_scale, self.prompt, self.negative_prompt, self.boost_face_quality, self.blending, self.callback_steps, self.show_patches, self.dummy_upscale)
            
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

# Upscale preview thread
class UpscalePreviewThread(QThread):
    preview_update_signal = pyqtSignal(object, int, str)  # Emits (current image, current tile, current iteration) during the upscaling process for gui progress bars

    def __init__(self):
        super().__init__()

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

    def visualize_patches(self, image):
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

        return image

    def update_preview_tile(self, patch_image, patch_number, image_path, show_patches, preview_image):
        image = Image.open(image_path)
        if show_patches:
            image = self.visualize_patches(image_path)

        self.preview_image = preview_image        
        window_size = 128
        min_padding_size = 8                      # Pixels of padding on right and bottom sides of the patches
        patch_size = window_size - min_padding_size        # Size of the patches to be extracted from the image in pixels
        input_image_width, input_image_height = image.size

        x_overlap, x_last_overlap, number_of_windows_in_row = self.calculate_dynamic_overlap(input_image_width, window_size, patch_size)
        y_overlap, y_last_overlap, number_of_windows_in_col = self.calculate_dynamic_overlap(input_image_height, window_size, patch_size)

        x_overlap = x_overlap * 4
        x_last_overlap = x_last_overlap * 4
        y_overlap = y_overlap * 4
        y_last_overlap = y_last_overlap * 4

        window_size = window_size * 4

        r = patch_number % number_of_windows_in_row
        c = patch_number // number_of_windows_in_row

        if r == number_of_windows_in_row - 1:
            x_start_point = (r * window_size) - (r * x_last_overlap * 2)
        else:
            x_start_point = (r * window_size) - (r * x_overlap * 2)

        if c == number_of_windows_in_col - 1:
            y_start_point = (c * window_size) - (c * y_last_overlap * 2)
        else:
            y_start_point = (c * window_size) - (c * y_overlap * 2)

        # add the patch image to the preview image in the correct location
        self.preview_image.paste(patch_image, (x_start_point, y_start_point))

        # Draw a border around the patchwith the colour genrated from the patch number
        colour = plt.cm.jet(patch_number / (number_of_windows_in_col * number_of_windows_in_row))
        # Convert the float values to integers for the color tuple
        colour_int = tuple(int(x * 255) for x in colour[:-1])
        #draw = ImageDraw.Draw(image)
        #draw.rectangle([x_start_point, y_start_point, x_start_point + window_size, y_start_point + window_size], outline=colour_int)

        return self.preview_image

    def create_plot_data(self, image_path, show_patches):
        image = Image.open(image_path)          
        if show_patches:
            image = self.visualize_patches(image_path)

        # saclae up image dimesions by 4x so that isd the right size to accept the upscaled tiles back in for comparison in the gui
        self.image = image.resize((image.width * 4, image.height * 4), Image.NEAREST)
        self.preview_image = self.image.copy()

        return self.image, self.preview_image

#%%  - Load the UI file
Form, Window = uic.loadUiType("SDx4_interface.ui")
app = QApplication([])


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

        ### Initialise UI Elements
        self.check_preferences()          # Checks the user preferences file to load the correct settings
        self.init_modularUI()           # Modular Unified Ui
        self.init_programUI()           # Main Program Ui elements
        self.initalise_settings()          # User Settings 

        self.upscale_preview_thread = UpscalePreviewThread()

    #%% - Initialise Ui Elements
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
        self.init_icons()
        self.set_theme()

        ### SETTINGS PAGE
        self.ui.themesListSelector.currentTextChanged.connect(self.set_theme)
        self.ui.checkForUpdates_ProgramBtnType.clicked.connect(self.check_online_for_updates)
        self.ui.downloadUpdateBtn_ProgramBtnType.clicked.connect(self.download_latest_version)

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
        self.ui.runUpscaleBtn.clicked.connect(self.run_upscale)

        # if a user has selcted an item in the list connect the itemClicked signal to the display image function
        self.ui.inputFilesListDisplay.itemClicked.connect(self.file_selected_in_list)

        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout for the widget_28 if it's not a layout already
        layout = QVBoxLayout(self.ui.widget_28)

        # Add the canvas to the layout
        layout.addWidget(self.canvas)

        # Plot some example data
        self.update_plot_data(None)   # remove path and set to None to start with !!!!!

    #%% - Modular Ui Functions
    def init_icons(self):
        
        self.iconw = QIcon()
        self.iconb = QIcon()
        self.iconw.addPixmap(QPixmap(":/whiteicons/icons/light/align-justify.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.iconb.addPixmap(QPixmap(":/blackicons/icons/dark/align-justify.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon1w = QIcon()
        self.icon1b = QIcon()
        self.icon1w.addPixmap(QPixmap(":/whiteicons/icons/light/home.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon1b.addPixmap(QPixmap(":/blackicons/icons/dark/home.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon2sw = QIcon()
        self.icon2sb = QIcon()
        self.icon2sw.addPixmap(QPixmap(":/whiteicons/icons/light/trending-up.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon2sb.addPixmap(QPixmap(":/blackicons/icons/dark/trending-up.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon2w = QIcon()
        self.icon2b = QIcon()
        self.icon2w.addPixmap(QPixmap(":/whiteicons/icons/light/printer.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon2b.addPixmap(QPixmap(":/blackicons/icons/dark/printer.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon3w = QIcon()
        self.icon3b = QIcon()
        self.icon3w.addPixmap(QPixmap(":/whiteicons/icons/light/settings.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon3b.addPixmap(QPixmap(":/blackicons/icons/dark/settings.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon4w = QIcon()
        self.icon4b = QIcon()
        self.icon4w.addPixmap(QPixmap(":/whiteicons/icons/light/info.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon4b.addPixmap(QPixmap(":/blackicons/icons/dark/info.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon5w = QIcon()
        self.icon5b = QIcon()
        self.icon5w.addPixmap(QPixmap(":/whiteicons/icons/light/help-circle.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon5b.addPixmap(QPixmap(":/blackicons/icons/dark/help-circle.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon6w = QIcon()
        self.icon6b = QIcon()
        self.icon6w.addPixmap(QPixmap(":/whiteicons/icons/light/x-circle.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon6b.addPixmap(QPixmap(":/blackicons/icons/dark/x-circle.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon7w = QIcon()
        self.icon7b = QIcon()
        self.icon7w.addPixmap(QPixmap(":/whiteicons/icons/light/more-horizontal.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon7b.addPixmap(QPixmap(":/blackicons/icons/dark/more-horizontal.svg"), QIcon.Mode.Normal, QIcon.State.Off)

        self.icon8w = QIcon()
        self.icon8b = QIcon()
        self.icon8w.addPixmap(QPixmap(":/whiteicons/icons/light/sun.svg"), QIcon.Mode.Normal, QIcon.State.Off)
        self.icon8b.addPixmap(QPixmap(":/blackicons/icons/dark/moon.svg"), QIcon.Mode.Normal, QIcon.State.Off)

    def set_icons_color(self, color_name="white"):
        self.ui.leftMenuBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.settingsBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.infoBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.helpBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.centerMenuCloseBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.notificationCloseBtn_UiBtnType.icon().changeColor(color_name)
        self.ui.cpuoffloadCheckbox.changeColor(color_name)
        """
        for attr in dir(self.ui):
            component = getattr(self.ui, attr)
            if isinstance(component, QPushButton):
                # check if the button has an image
                if component.icon().isNull() == False: # if the button has an image
                    component.icon().changeColor(color_name)
                    break
        """

    def set_icons_white(self):
        self.ui.leftMenuBtn_UiBtnType.setIcon(self.iconw)
        self.ui.settingsBtn_UiBtnType.setIcon(self.icon3w)
        self.ui.infoBtn_UiBtnType.setIcon(self.icon4w)
        self.ui.helpBtn_UiBtnType.setIcon(self.icon5w)
        self.ui.centerMenuCloseBtn_UiBtnType.setIcon(self.icon6w)
        self.ui.notificationCloseBtn_UiBtnType.setIcon(self.icon6w)
        self.ui.cpuoffloadCheckbox.changeColor("white")

    def set_icons_black(self):
        self.ui.leftMenuBtn_UiBtnType.setIcon(self.iconb)
        self.ui.settingsBtn_UiBtnType.setIcon(self.icon3b)
        self.ui.infoBtn_UiBtnType.setIcon(self.icon4b)
        self.ui.helpBtn_UiBtnType.setIcon(self.icon5b)
        self.ui.centerMenuCloseBtn_UiBtnType.setIcon(self.icon6b)
        self.ui.notificationCloseBtn_UiBtnType.setIcon(self.icon6b)
        self.ui.cpuoffloadCheckbox.changeColor("black")

    def set_helpandinfo_copy(self, help_file_path, info_file_path):
        with open(help_file_path, "r") as file:
            self.ui.helpTextCopy.setText(self.ui.helpTextCopy.toHtml().replace("Help file failed to load. Sorry for the mix-up, we are scrambling to fix it!", file.read()))

        with open(info_file_path, "r") as file:
            self.ui.infoTextCopy.setText(self.ui.infoTextCopy.toHtml().replace("Information file failed to load. Sorry for the mix-up, we are scrambling to fix it!", file.read()))

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
        self.boost_face_quality = True   # THSI CURRENTLY OVERIDES THE LOADED SETTING BUT IM USING FOR DEBUGGING!! , REMOVE WHEN DONE

        self.processing_file_list = []

        self.displayed_file = None

    def run_animation(self, animation_item, start, end):
        animation_item.setStartValue(start)
        animation_item.setEndValue(end)
        animation_item.start()

    def expandorshrink_left_menu(self):
        # If the left menu is closed (width = 40), then it is opened
        if self.ui.leftMenuContainer.maximumWidth() == 50:
            self.run_animation(self.left_menu_animation, start=50, end=250)
        else:
            self.run_animation(self.left_menu_animation, start=250, end=50)

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

    def switch_ui_mode(self):
        if self.current_ui_mode == "light":
            with open(self.dark_mode_path, "r") as file:
                self.setStyleSheet(file.read())
            self.current_ui_mode = "dark"
            #self.set_icons_white()
            self.set_icons_color("white")

        elif self.current_ui_mode == "dark":
            with open(self.light_mode_path, "r") as file:
                self.setStyleSheet(file.read())
            self.current_ui_mode = "light"
            #self.set_icons_black()
            self.set_icons_color("black")

    def set_theme(self):
        '''Sets the theme of the UI based on the selected theme in the themesListSelector'''

        # Get the selected theme from the themesListSelector
        selected_theme = self.ui.themesListSelector.currentText()

        # Set the current theme to the selected theme
        self.current_theme = selected_theme

        self.dark_mode_path = resource_path(fr"App_Data\themes\{self.current_theme}\dark_mode.css")
        self.light_mode_path = resource_path(fr"App_Data\themes\{self.current_theme}\light_mode.css")

        #switch to the new css based on the previos mode i.e dark/light
        if self.current_ui_mode == "dark":
            with open(self.dark_mode_path, "r") as file:
                self.setStyleSheet(file.read())
        else:
            with open(self.light_mode_path, "r") as file:
                self.setStyleSheet(file.read())

        # Save the current theme as default startup theme in the config JSON file
        with open(self.config_file, 'r') as file:
            data = json.load(file)

        data["Theme"] = self.current_theme   # Update the "Theme" value

        with open(self.config_file, 'w') as file:
            json.dump(data, file)  

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
        self.ui.inputFilesListDisplay.addItems(file_paths)

        for file_path in file_paths:
            self.add_file_to_processing_list(file_path)

    def browse_input_folders(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directories")
        if folder_path:              # if user selects a path wothout canceling the dialog box
            self.ui.inputFilesListDisplay.addItems(os.listdir(folder_path))     # add the filenames to the display list 

            for file in os.listdir(folder_path):                          # add full file paths to the processing file list
                self.add_file_to_processing_list(os.path.join(folder_path, file))

    def add_file_to_processing_list(self, file_path):
        self.temp_folder = r"App_Data\temp_data"   # move to main section of init of main class

        # copy the file to the temp folder
        shutil.copy(file_path, self.temp_folder)

        # get the new path in the temp folder
        file_name = os.path.basename(file_path)
        file_path = os.path.join(self.temp_folder, file_name)

        # duplicate the file in the temp folder and add _preview to the end of the file name
        preview_file_name = file_name.replace(".", "_preview.")
        preview_file_path = os.path.join(self.temp_folder, preview_file_name)
        shutil.copy(file_path, preview_file_path)

        # add full file paths to the processing file list
        file_info_dict = {}
        file_info_dict["input_file_path"] = file_path
        file_info_dict["preview_file_path"] = preview_file_path
        self.processing_file_list.append(file_info_dict)

    def remove_selected_list_item(self):
        # if a user has selcted an item in the list then remove it from the list and the processing file list and remove its images from the temp folder 
        if self.ui.inputFilesListDisplay.currentItem():
            self.ui.inputFilesListDisplay.takeItem(self.ui.inputFilesListDisplay.currentRow())
            self.processing_file_list.pop(self.ui.inputFilesListDisplay.currentRow())
            self.displayed_file = None
            self.update_plot_data(None)

    def file_selected_in_list(self):
        # If a file is selected in the list, set self.diplayed_file to the full file path of the selected file
        if self.ui.inputFilesListDisplay.currentItem():
            self.displayed_file = self.ui.inputFilesListDisplay.currentItem().text()

        # if the displayed file is not None then display it in the image viewer
        if self.displayed_file:
            self.update_plot_data(self.displayed_file)


    def run_upscale(self):
        print("Running upscale")
        print("Blending:", self.blending)
        print("Blend Mode:", self.blend_mode)
        print("Boost Face Quality:", self.boost_face_quality)
        print("CPU Offload:", self.cpu_offload)
        print("Attention Slicing:", self.attention_slicing)
        print("Xformers:", self.xformers)
        print("Output Directory:", self.output_dir)
        print("Number of Inference Steps:", self.num_inference_steps)
        print("Guidance Scale:", self.guidance_scale)
        print("Processing File List:", self.processing_file_list)

        if self.ui.makeSettingsDefaultCheckbox.isChecked():
            self.update_defaults()

        # disable all ui buttons from interaction other than the run upscale button
            
        # change the run upscale button text to "Stop Upscale"
            
        # Change the run upscale button to instead run the stop upscale function (perhaps instead of doing htis here just have the function the button is connected to check if the upscale is running and then adjust itts behaviour to either run or stop the thred)

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


    def update_plot_data(self, image_path):
        if image_path == None:                                  # show no plot 
            self.ax.clear()
            # add text on the plot to tell the user to select an image
            self.user_text_warning = self.ax.text(0.5, 0.5, 'Please select an image', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=20)

        else:
            self.user_text_warning.set_text('')                 # remove the user text warning

            self.image, self.preview_image = self.upscale_preview_thread.create_plot_data(image_path, self.show_patches)
            self.ax.imshow(self.preview_image, cmap='viridis')
            self.figure.tight_layout()

        self.ax.axis('off')                                 # turn off axis
        self.figure.patch.set_alpha(0)                      # set mpl background alpha to 0 to make it transparent 
        self.canvas.draw()                 # Refresh the canvas

    def add_patch_ontopof_image(self, patch_image, patch_number, image_path):
        self.preview_image = self.upscale_preview_thread.update_preview_tile(patch_image, patch_number, image_path, self.show_patches, self.preview_image)
        self.ax.imshow(self.preview_image)
        
        self.ax.axis('off')
        self.figure.tight_layout()
        self.figure.patch.set_alpha(0)

        # Refresh the canvas
        self.canvas.draw()

    #%% - Backend  Functions
    def start_file_processing_thread(self):
        
        # Initialise Backend
        self.file_processing_thread = FileProcessingThread( self.processing_file_list,              # specify list of input images
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
                                                            self.safety_checker
                                                        )

        # Connect signals from the file processing thread to gui
        self.file_processing_thread.finished.connect(self.file_processing_finished)
        self.file_processing_thread.stopped.connect(self.file_processing_stopped)
        self.file_processing_thread.processing_position_signal.connect(self.update_progress_bars)
        self.file_processing_thread.patch_preview.connect(self.add_patch_ontopof_image)
        self.file_processing_thread.tile_complete_signal.connect(self.add_patch_ontopof_image)
        
        # Start the file processing thread
        self.file_processing_thread.start()

    def cancel_file_processing(self):
        # Request interruption to stop the file processing thread
        self.file_processing_thread.requestInterruption()

    def file_processing_finished(self):
        print("File processing finished")
        # Add any additional actions you want to perform when processing is finished

    def file_processing_stopped(self):
        print("File processing stopped")
        # Add any additional actions you want to perform when processing is stopped

    def update_progress_bars(self, image_num, total_images, tile_num, total_tiles, iteration_num, total_iterations):

        # Calulate the percentage of images, tiles and iterations processed
        image_percentage = int((image_num / total_images) * 100)            
        tile_percentage = int((tile_num / total_tiles) * 100)            
        iteration_percentage = int((iteration_num / total_iterations) * 100)          
        
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



