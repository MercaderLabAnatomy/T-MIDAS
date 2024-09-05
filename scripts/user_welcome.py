import subprocess
import os 
import tkinter as tk
from tkinter import filedialog
import csv
import datetime
now = datetime.datetime.now()
import textwrap
wrapper = textwrap.TextWrapper(width=80)
date = now.strftime("%Y-%m-%d %H:%M:%S")

import warnings

# Ignore all warnings
warnings.simplefilter('ignore')


"""
Description: This script is the main menu for the Tissue Microscopy Image Data Analysis Suite (T-MIDAS).

The script provides a user-friendly interface to run different image processing pipelines.

"""



# if T-MIDAS is not in /opt, ask user where it is
if not os.path.exists('/opt/T-MIDAS'):
    print("T-MIDAS is not in /opt. Please provide the path to the T-MIDAS folder.")
    # make a popup window appear to ask for the path to T-MIDAS
    tmidas_path = filedialog.askdirectory(title="Please provide the path to the T-MIDAS folder.")
    os.environ["TMIDAS_PATH"] = tmidas_path
    print("T-MIDAS path set to " + os.environ["TMIDAS_PATH"])
else:
    os.environ["TMIDAS_PATH"] = "/opt/T-MIDAS" 

    




def get_available_RAM():
    return subprocess.check_output("free -h | grep -E 'Mem:' | awk '{print $2}'", 
                                   shell=True).decode('utf-8').strip().replace("Gi", "") + " GB"

def get_model_name_CPU():
    return subprocess.check_output("lscpu | grep -E 'Model name'", 
                                   shell=True).decode('utf-8').strip().replace("Model name:", "").replace(" ", "")

def get_hostname():
    return subprocess.check_output("hostname", 
                                   shell=True).decode('utf-8').strip()


def get_no_threads():
    return subprocess.check_output("lscpu | grep -E '^CPU\(s\):'", 
                                   shell=True).decode('utf-8').strip().replace(" ", "").replace("CPU(s):", "")

def get_model_name_GPU():
    return subprocess.check_output("nvidia-smi --query-gpu=gpu_name --format=csv,noheader", 
                                             shell=True).decode('utf-8').strip()

def get_available_VRAM():
    available_RAM_GPU = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", 
                                                shell=True).decode('utf-8').strip()
    return str(round(int(available_RAM_GPU) / 1024, 2)) + " GB"


def welcome_message():
    os.system('clear')
    print('''
╔╦╗  ╔╦╗╦╔╦╗╔═╗╔═╗
 ║───║║║║ ║║╠═╣╚═╗
 ╩   ╩ ╩╩═╩╝╩ ╩╚═╝          
          ''')
    print(f" Welcome to the Tissue Microscopy Image Data Analysis Suite! \n")
    print(f" This machine with the name {get_hostname()} has\n") 
    print(f"- a {get_model_name_CPU()} CPU with {get_no_threads()} threads, and\n")
    print(f"- a {get_model_name_GPU()} GPU. \n") 
    print(f" Currently, {get_available_RAM()} RAM and {get_available_VRAM()} VRAM are available. \n")

    global user_name
    if 'user_name' not in globals():
        user_name = input("\n What's your name? ")

    main_menu() # go straight ahead into the image processing menu

def logging(env_name, script_name, input_parameters=None, user_name=None):
    # get input folder from input_parameters
    input_folder = input_parameters.split(' ')[1] # this relies on input_folder being the first parameter
    print(f"\n{user_name}, I am logging your choices to " + f'{input_folder}/{env_name}_log.csv')
    with open(f'{input_folder}/{env_name}_log.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([date, env_name, script_name, input_parameters, user_name])


def python_script_environment_setup(env_name, script_name, input_parameters=None):
    print("\n")
    print(f"\nRunning chosen pipeline with the following parameters: \n{input_parameters}")
    print("\n")
    subprocess.run(f"mamba run --live-stream -n {env_name} python {script_name} {input_parameters}".split(),
                   capture_output=False,text=True,cwd="/mnt/")
    print("\nDone.")
    logging(env_name, script_name, input_parameters, user_name)
    restart_program()
    
    
def popup_input(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = filedialog.askdirectory(title=prompt, initialdir="/mnt/")
    return user_input




def main_menu():
    os.system('clear')
    print(f"\nHi {user_name}! What would you like to do?\n")
    print("[1] Image Preprocessing")
    print("[2] Image Segmentation")
    print("[3] Regions of Interest (ROI) Analysis")
    print("[4] Image Segmentation Validation")
    print("[n] Start Napari")
    print("[x] Exit \n")
    
    choice = input("\nEnter your choice: ")
    
    
    if choice == "1":
        image_preprocessing()
        restart_program()
    if choice == "2":
        image_segmentation()
        restart_program()      
    if choice == "3":
        ROI_analysis()
        restart_program()
    if choice == "4":
        validation()
        restart_program()
    if choice == "n" or choice == "N":
        start_napari()
        restart_program()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()
    

def start_napari():
    os.system('clear')
    print("\nStarting Napari. Please wait...")
    # start napari in tmidas-env
    subprocess.run(f"mamba run --live-stream -n tmidas-env napari".split(),
                   capture_output=False,text=True,cwd="/mnt/")
    print("\nNapari closed.")
    restart_program()


def image_preprocessing():
    
    os.system('clear')
    print("Image Preprocessing: What would you like to do?\n")
    print("[1] File Conversion to TIFF")
    print("[2] Cropping Largest Objects from Images")
    print("[3] Extract intersecting regions of two images")
    print("[4] Sample Random Image Subregions")
    print("[5] Enhance contrast of single color image using CLAHE")
    print("[6] Restore images using Cellpose")
    print("[7] Split color channels (2D or 3D, also time series)")
    print("[8] Merge color channels (2D or 3D, also time series)")
    print("[9] Convert RGB images to label images")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")

    choice = input("\nEnter your choice: ")
    
    
    if choice == "1":
        file_conversion()
        restart_program()
    if choice == "2":
        crop_images()
        restart_program()
    if choice == "3":
        os.system('clear')
        print(wrapper.fill("Extract intersecting regions of two images: A popup will appear in a moment asking you to select the folder containing the images. Image sets will be discriminated by unique suffixes. You will be asked to enter the suffixes of the two sets images you want to intersect. The first set of images will be used as a mask to extract the intersecting regions from the second set of images."))
        input_folder = popup_input("\nEnter the path to the folder containing the images: ")
        maskfiles = input("\nEnter the suffix of mask images (example: _labels.tif): ")
        intersectfiles = input("\nEnter suffix of images to be intersected: ")
        output_tag = input("\nEnter the suffix of the output images: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/intersection.py',
                                        '--input ' + input_folder + 
                                        ' --maskfiles ' + maskfiles + 
                                        ' --intersectfiles ' + intersectfiles +
                                        ' --output_tag ' + output_tag)
        restart_program()
    if choice == "4":
        os.system('clear')
        print('''
              \nRandom Tile Sampling: If your image consists of multiple channels, 
              please provide all channels in a single multichannel .tif image.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the (multichannel) .tif images: ")
        tile_diagonal = input("\nEnter the tile diagonal in pixels: ")
        percentage = input("\nEnter the percentage of random tiles to be picked from the entire image (20-100): ")
        random_seed = input("\nEnter a random seed for reproducibility (integer): ")    
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/random_tile_sampler.py',
                                        '--input ' + input_folder + 
                                        ' --tile_diagonal ' + tile_diagonal + 
                                        ' --percentage ' + percentage +
                                        ' --random_seed ' + random_seed)
        restart_program()
    if choice == "5":
        os.system('clear')
        print(wrapper.fill("You chose to apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to single color images. A popup will appear in a moment asking you to select the folder containing the .tif images. You will be asked to enter a few parameter values. Default values:"))
        print("\n")
        print(wrapper.fill("- kernel_size: Should correspond to the size of the features that should be enhanced (e.g. 64 for 64x64 pixel tiles or 64x64x64 pixel volumes),"))
        print("\n")
        print(wrapper.fill("- nbins <= 256 (corresponds to color depth of 8bit images)"))
        print("\n")
        print(wrapper.fill("- clip_limit: 0.01 (> 1/nbins, controls extent of contrast enhancement)"))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        kernel_size = input("\nEnter the kernel size: ")
        clip_limit = input("\nEnter the clip limit: ")
        nbins = input("\nEnter the number of bins: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/deep_tissue_clahe.py',
                                        '--input ' + input_folder + ' --kernel_size ' + kernel_size + ' --clip_limit ' + clip_limit + ' --nbins ' + nbins)
        restart_program()

    if choice == "6":
        os.system('clear')
        print(wrapper.fill("You chose to restore images using Cellpose. A popup will appear in a moment asking you to select the folder containing single color channel .tif images. They can be 2D, z-stack or time series. Single color channel is recommended for speed (You can split multicolor images with T-MIDAS). You will be asked whether you want to denoise or deblur. Lastly, you will be asked for the dimension order of the images. For example TZYX (frames, slices, height, width), but this can vary depending on how the image was acquired or processed. In Python, you can find out by using imread and checking the shape of the image."))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        # num_channels= input("\nEnter the number of color channels to be restored: ")
        restoration_type = input("\nChoose between denoising (dn), deblurring (db) or upsampling (us): ")
        object_type = input("\nChoose between nuclei (n) or cytoplasm (c): ")
        dim_order = input("\nEnter the dimension order of the images (example: TZYX): ")
        num_channels = input("\nEnter the number of color channels (default=1): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/restore_cellpose.py',
                                        '--input ' + input_folder + #' --num_channels ' + num_channels + 
                                        ' --restoration_type ' + restoration_type + 
                                        ' --object_type ' + object_type + ' --dim_order ' + dim_order + ' --num_channels ' + num_channels)

        # + ' --object_type ' + object_type) 
        restart_program()


    if choice == "7":
        os.system('clear')
        print(wrapper.fill("You chose to split the color channels of multicolor images. A popup will appear in a moment asking you to select the folder containing the multicolor images. You will be asked to enter the dimension order of the images and the names of the color channel output folders."))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the multicolor images: ")
        dim_order = input("\nEnter the dimension order of the images (example: XYZCT): ")
        channels = input("\nEnter the names of the color channels (example: FITC DAPI TRITC): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/split_color_channels.py',
                                        '--input ' + input_folder + ' --dim_order ' + dim_order + ' --channels ' + channels)
        restart_program()
        
    if choice == "8":
        os.system('clear')
        print(wrapper.fill("You chose to merge the color channels of multicolor images. A popup will appear in a moment asking you to select the folder containing the color channel folders. You will be asked to enter the names of the color channels."))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the color channel folders: ")
        channel_names = input("\nEnter the names of the color channels (example: FITC DAPI TRITC): ")
        dim_order = input("\nEnter the dimension order of the images (example: XYZCT): ")
        # use_gpu = input("\nUse GPU for processing? May terminate if images are too large (y/n): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/merge_color_channels.py',
                                        '--input ' + input_folder + ' --channels ' + channel_names + 
                                        ' --dim_order ' + dim_order)# + ' --gpu ' + use_gpu)
        restart_program()
    if choice == "9":
        os.system('clear')
        print('''You chose to convert RGB .tif files to .tif label images. \n
              A popup will appear in a moment asking you to select the folder containing the RGB .tif files.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the RGB .tif files: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/RGB_2_labels.py',
                                        '--folder ' + input_folder)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()

def file_conversion():
    os.system('clear') 
    print("\nFile Conversion to TIFF: Which file format would you like to convert?\n")
    print("[1] Convert .ndpi")
    print("[2] Convert bioformats-compatible series images (.lif, .czi, ...)")
    print("[3] Convert brightfield .czi")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        os.system('clear')
        print('''You chose to convert multicolor .ndpi files to .tif files. \n
              A popup will appear in a moment asking you to select the folder containing the .ndpi(s) files.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi(s) files: ")
        LEVEL = input('''
                      \nEnter the desired resolution level 
                      (0 = highest resolution, 1 = second highest resolution): 
                      ''')
        print('''\nConverting .ndpi files to .tif files. 
              Beware: If the image is too large this will raise an exception. 
              In that case, better use the ndpi file cropping option [1][2] to extract your regions of interest from the slide.
              ''')
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ndpis_to_tifs.py',
                                        '--input ' + input_folder + ' --level ' + LEVEL)
        restart_program()
    if choice == "2":
        os.system('clear')
        print('''You chose to extract .tif files from bioformats-compatible series images (.lif, .czi, ...).\n
              A popup will appear in a moment asking you to select the folder containing the series images. 
              Series of each image will be exported as .tif files with resolution metadata.''')
        input_folder = popup_input("\nEnter the path to the folder containing the series images: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/extract_tifs_from_series.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "3":
        os.system('clear')
        print('''You chose to convert brightfield .czi files to .tif files. \n
              A popup will appear in a moment asking you to select the folder containing the brightfield .czi(s) files.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the brightfield .czi(s) files: ")
        scale_factor = input("\nEnter the scale factor (0.5 = half the size (default)): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/czi_to_tif_brightfield.py',
                                        '--input ' + input_folder + 
                                        ' --scale_factor ' + scale_factor) # whitespace is important here
        restart_program()

    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()


def crop_images():
    os.system('clear')    
    print("\nImage Cropping:\n")
    print("[1] Slidescanner images (fluorescent, .ndpi)")
    print("[2] Slidescanner images (brightfield, .ndpi)")
    print("[3] Multicolor image stacks (.lif)")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")

    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        os.system('clear')
        print('''You chose to crop blobs from multicolor .ndpi files.  \n
              A popup will appear in a moment asking you to select the folder containing ndpi(s) files.''')
        input_folder = popup_input("\nEnter the path to the folder containing ndpi(s) files: ")
        padding = input("\nEnter the padding in pixels (default: 10): ")
        CROPPING_TEMPLATE_CHANNEL_NAME = input('''
                                               Enter the channel name 
                                               that represents the cropping template 
                                               (for example FITC or CY5): 
                                               ''')
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ndpis_to_cropped_tifs.py',
                                        '--input ' + input_folder + 
                                        ' --padding ' + padding +
                                        ' --cropping_template_channel_name ' + CROPPING_TEMPLATE_CHANNEL_NAME)
        restart_program()            

    if choice == "2":
        print('''You chose to crop blobs from brightfield .ndpi files. \n
              A popup will appear in a moment asking you to select the folder containing ndpi files.''')
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi files: ")
        padding = input("\nEnter the padding in pixels (default: 10): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ndpis_to_cropped_tifs_brightfield.py',
                                        '--input ' + input_folder +
                                        ' --padding ' + padding)
        restart_program()
    if choice == "3":
        os.system('clear')
        print('''You chose to crop blobs from multicolor .lif files. \n
              A popup will appear in a moment asking you to select the folder containing ndpi files.''')
        input_folder = popup_input("\nEnter the path to the .lif file: ")
        template_channel = input('''
                                 Enter the channel number 
                                 that represents the cropping template 
                                 (single channel: 0): 
                                 ''')
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/lif_to_cropped_tifs.py',
                                        '--input_folder ' + input_folder + ' --template_channel ' + template_channel)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()


def image_segmentation():
    os.system('clear')
    print("\nImage Segmentation: What would you like to do?\n")
    print("[1] Segment bright spots (2D or 3D, also time series)")
    print("[2] Segment blobs (2D or 3D, also time series)")
    # print("[3] Segment blobs (3D; requires dark background and good SNR)")
    print("[3] Semantic segmentation (2D, fluorescence or brightfield)")
    print("[4] Semantic segmentation (2D; Segment Anything)")   
    print("[5] Semantic segmentation (3D; requires dark background and good SNR)")
    print("[6] Improve instance segmentation using CLAHE")
    # print("[7] Segment multicolor images of cell cultures (2D)")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    
   
    if choice == "1":
        os.system('clear')
        print('''You chose to segment bright spots in 2D. \n
                A popup will appear in a moment asking you to select the folder containing the intensity images.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the intensity images: ")
        dim_order = input("\nEnter the dimension order of the images (example: TZYX): ")
        bg = input("\nWhat kind of background? (1 = gray, 2 = dark): ")
        if bg == "1":
            print("\n")
            print(wrapper.fill('''You chose to segment bright spots with background. You will be asked to enter an intensity threshold.'''))
            intensity_threshold = input("\nEnter the intensity threshold (0 < value < 255): ")
            python_script_environment_setup('tmidas-env', 
                                  os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_spots.py',
                                  '--input ' + input_folder + ' --dim_order ' + dim_order +
                                  ' --bg ' + bg + ' --intensity_threshold ' + intensity_threshold)
        if bg == "2":
            print("\n")
            print(wrapper.fill('''You chose to segment bright spots with dark background.'''))
            python_script_environment_setup('tmidas-env', 
                                  os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_spots.py',
                                  '--input ' + input_folder + ' --dim_order ' + dim_order + ' --bg ' + bg)
        else:
            print("Invalid choice")
            restart_program()


        restart_program()
        
    if choice == "2":
        os.system('clear')
        print("\n")
        print("---------------------------------")
        print("You chose to segment blobs in 2D.")
        print("---------------------------------")
        print("\n")
        print(wrapper.fill('''A popup will appear in a moment asking you to select the folder containing the .tif images.'''))
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        print("\n")
        print(wrapper.fill("You can choose between two methods:"))
        print("\n")
        print(wrapper.fill("[1] User-defined or automatic (Otsu) intensity thresholding."))
        print("\n")
        print(wrapper.fill("[2] Cellpose's (generalist) cyto3 model."))
        print("\n")
        choice = input("\nEnter your choice: ")
        if choice == "1":
            print("\nYou chose classical gauss-otsu labeling.")
            threshold = input("\nEnter an intensity threshold value within in the range 1-255 if you want to define it yourself or enter 0 to use automatic thresholding: ")
            use_filters = input("\nUse filters for user-defined segmentation? (yes/no): ")
            exclude_small = input("\nLower size threshold to exclude small objects: ")
            exclude_large = input("\nUpper size threshold to exclude large objects: ")
            dim_order = input("\nEnter the dimension order of the images (example: TZYX): ")
            python_script_environment_setup('tmidas-env', 
                                            os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_blobs.py',
                                            '--input ' + input_folder + 
                                            ' --threshold ' + threshold +
                                            ' --use_filters ' + use_filters +
                                            ' --exclude_small ' + exclude_small + 
                                            ' --exclude_large ' + exclude_large +
                                            ' --dim_order ' + dim_order)
            restart_program()
        if choice == "2":
            print("\nYou chose Cellpose's cyto3 model.")
            print("\n")
            print(wrapper.fill("First, you will be asked to enter the typical diameter of the objects that you want to segment. If you want to let Cellpose predict the diameter, enter 0. Diameter prediction only works for 2D images."))
            print("\n")
            diameter = input("\nEnter the typical diameter of the objects that you want to segment: ")
            print("\n")
            print(wrapper.fill("Next, you will be asked to enter the channels to use. Gray=0, Red=1, Green=2, Blue=3. Single (gray) channel, enter 0 0. For green cytoplasm and blue nuclei, enter 2 3."))
            print("\n")
            channels = input("\nEnter the channels to use (example: 0 0):")
            dim_order = input("\nEnter the dimension order of the images (example: TZYX): ")
            python_script_environment_setup('tmidas-env', 
                                            os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_blobs_cyto3.py',
                                            '--input ' + input_folder +
                                            ' --diameter ' + diameter +
                                            ' --channels ' + channels +
                                            ' --dim_order ' + dim_order)
            restart_program()

    
    # if choice == "3":
    #     os.system('clear')
    #     print('''You chose to segment blobs in 3D. \n
    #           A popup will appear in a moment asking you to select the folder containing the .tif images.
    #           ''')
    #     input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
    #     nuclei_channel = input("\nEnter number of the color channel you want to segment:  ")
    #     python_script_environment_setup('tmidas-env', 
    #                                     os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_instances_3D.py',
    #                                     '--image_folder ' + input_folder + ' --nuclei_channel ' + nuclei_channel)
    #     restart_program()
    if choice == "3":
        os.system('clear')
        print('''You chose semantic segmentation (2D, fluorescence or brightfield).\n
                A popup will appear in a moment asking you to select the folder containing the .tif images.
                
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        image_type = input("\nBrightfield images? (y/n): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_semantic_2D.py',
                                        '--input ' + input_folder + ' --image_type ' + image_type)
        restart_program()
    if choice == "4":
        os.system('clear')
        print('''You chose semantic segmentation (2D; Segment Anything).\n
                A popup will appear in a moment asking you to select the folder containing the .tif images.
                
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_semantic_SAM_2D.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "5":
        os.system('clear')
        print('''You chose semantic segmentation (3D). \n
                A popup will appear in a moment asking you to select the folder containing the .tif images.
                ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        #tissue_channel = input("\nEnter number of the color channel you want to segment: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_semantic_3D.py',
                                        '--image_folder ' + input_folder)# + ' --tissue_channel ' + tissue_channel)
        restart_program()
    if choice == "6":
        os.system('clear')
        print(wrapper.fill("You chose to improve instance segmentations using CLAHE. A popup will appear in a moment asking you to select the folder containing the .tif images. You will be asked to enter a few parameter values. Default values:"))
        print("\n")

        print("\n")
        print(wrapper.fill("- outline_sigma: Defines the sigma for the gauss-otsu-labeling. Also typically in the range of 1.0-2.0."))
        print("\n")
        print(wrapper.fill("- kernel_size: Should correspond to the size of the features that should be enhanced (e.g. 64 for 64x64 pixel tiles or 64x64x64 pixel volumes),"))
        print("\n")
        print(wrapper.fill("- nbins <= 256 (corresponds to color depth of 8bit images)"))
        print("\n")
        print(wrapper.fill("- clip_limit: 0.01 (> 1/nbins, controls extent of contrast enhancement)"))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        mask_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        label_pattern = input("\nEnter the label image suffix. Example: *_labels.tif: ")
        kernel_size = input("\nEnter the kernel size: ")
        nbins = input("\nEnter the number of bins: ")
        clip_limit = input("\nEnter the clip limit: ")
        outline_sigma = input("\nEnter the outline sigma: ")
        exclude_small = input("\nLower size threshold to exclude small objects (example: 25.0): ")
        exclude_large = input("\nUpper size threshold to exclude large objects (example: 2500.0): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_instances_clahe.py',
                                        '--input ' + input_folder +
                                        ' --masks ' + mask_folder +
                                        ' --label_pattern ' + label_pattern +
                                        ' --kernel_size ' + kernel_size +
                                        ' --nbins ' + nbins +
                                        ' --clip_limit ' + clip_limit +
                                        ' --outline_sigma ' + outline_sigma +
                                        ' --exclude_small ' + exclude_small +
                                        ' --exclude_large ' + exclude_large
                                        )
        restart_program()
    # if choice == "7":
    #     os.system('clear')
    #     print('''You chose to segment multicolor images of cell cultures in 2D. \n
    #             A popup will appear in a moment asking you to select the folder containing the .tif images.
    #             ''')
    #     input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
    #     channels = input("\nEnter the names of the channels in the order they appear in the multicolor image (example: DAPI GFP RFP): ")
    #     tile_diagonal = input("\nEnter the tile diagonal in pixels: ")
    #     percentage = input("\nEnter the percentage of random tiles to be picked from the entire image (20-100): ")
    #     random_seed = input("\nEnter a random seed for reproducibility (integer): ")
    #     python_script_environment_setup('tmidas-env', 
    #                                     os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_multicolor_cell_culture_2D.py',
    #                                     '--input ' + input_folder +
    #                                     ' --channels ' + channels +
    #                                     ' --tile_diagonal ' + tile_diagonal +
    #                                     ' --percentage ' + percentage +
    #                                     ' --random_seed ' + random_seed
    #                                     )                                       
    #     restart_program()

    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()



def ROI_analysis():
    os.system('clear')
    print("\nRegions of Interest (ROI) Analysis: What would you like to do?\n")
    print("[1] Heart slices: Add 100um boundary zone to [intact+injured] ventricle masks")
    print("[2] Count spots within ROI (2D)")
    print("[3] Count blobs within ROI (3D)")
    print("[4] Count Colocalization of ROI in 2 or 3 color channels")
    print("[5] Get properties of objects within ROI (two channels)")
    print("[6] Get basic ROI properties (single channel)")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    

    if choice == "1":
        os.system('clear')
        print('''You chose to create ROI from label images containing ventricle + injury masks. \n
                A popup will appear in a moment asking you to select the folder containing the label images.
                ''')
        input_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        pixel_resolution = input("\nEnter the pixel resolution of the images in um/px: ")
        intact_label_id = input("\nEnter the label id of the intact myocardium: ")
        injury_label_id = input("\nEnter the label id of the injury region: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/create_ventricle_ROI.py',
                                        '--input ' + input_folder + 
                                        ' --pixel_resolution ' + pixel_resolution +
                                        ' --intact_label_id ' + intact_label_id +
                                        ' --injury_label_id ' + injury_label_id)
        restart_program()
    if choice == "2":
        os.system('clear')
        print('''You chose to count spots in 2D ROI. You will have to provide two sets of label images: \n
                1. The label images containing the ROI (suffix: _labels.tif) and \n
                2. The label images containing the spots (suffix: _labels.tif). \n
                A popup will appear in a moment asking you to select the folder containing the label images.
                ''')

        input_folder = popup_input("\nInput: Folder with all label images (ROI and spots).")
        pixel_resolution = input("\nEnter the pixel resolution of the images in um/px: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ROI_count_instances_2D.py',
                                        '--input ' + input_folder + ' --pixel_resolution ' + pixel_resolution)
        restart_program() 
    if choice == "3":
        os.system('clear')
        print('''You chose to count blobs in 3D ROI. You will have to provide two sets of label images: \n
                1. The label images containing the ROI (suffix: _ROI_labels.tif) and \n
                2. The label images containing the blobs (suffix: _blob_labels.tif). \n
                Two popups will appear in a moment asking you to select the folders containing the label images.
                ''')
        input_folder = popup_input("\nEnter the path to the folder containing blob and ROI label image subfolders: ")
        blob_folder = input("\nEnter the name of the folder containing blob label images: ")
        ROI_folder = input("\nEnter the name of the folder containing ROI label images: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ROI_count_instances_3D.py',
                                        '--input_folder ' + input_folder +
                                        '--blob_folder ' + blob_folder + 
                                        ' --ROI_folder ' + ROI_folder)
        restart_program()
    if choice == "4":
        os.system('clear')
        print("\n")
        print("------------------------------------------------")
        print("You chose to colocalize ROI in different color channels.")
        print("------------------------------------------------")
        print("\n")
        print(wrapper.fill("""Input data structure: A popup will appear in a moment asking you to select the parent folder containing subfolders for each color channel. Those should contain the segmentations (label images with suffix _labels.tif). You will be asked to enter the names of all color channel folders. Please enter them in the order in which you want to colocalize them. Example: FITC DAPI TRITC would mean you want to count DAPI in FITC and TRITC in DAPI and FITC. Then enter the suffix of the label images of each channel in the same order. Example: *_labels.tif *_labels.tif *_labels.tif. 
                           """))
        print("\n")

        input_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        channels = input("\nEnter the names of your color channel subfolders in the abovementioned order (example: FITC DAPI TRITC): ")
        label_patterns = input("\nEnter the label patterns (example: *_labels.tif *_labels.tif *_labels.tif): ")
        #add_intensity = input("\nDo you want to quantify average intensity of the last channel in the ROI of the second last channel? (y/n): ")
        # output_images = input("\nDo you want to save colocalization images? (y/n): ")
        get_area = input("\nDo you want to get the area of the ROI of the first channel? (y/n): ")
        python_script_environment_setup('tmidas-env', 
                                    os.environ.get("TMIDAS_PATH")+'/scripts/ROI_colocalization_count_multicolor.py',
                                    '--input ' + input_folder +
                                    ' --channels ' + channels +
                                    ' --label_patterns ' + label_patterns +
                                    # ' --output_images ' + output_images +
                                    ' --get_area ' + get_area
                                    )
                                    #' --add_intensity ' + add_intensity
                                    
        restart_program()

    if choice == "5":
        os.system('clear')
        print("\n")
        print("----------------------------------------------------")
        print("You chose to get properties of objects within ROI.")
        print("----------------------------------------------------")
        print("\n")
        print(wrapper.fill('''A popup will appear in a moment asking you to select the parent folder containing subfolders for each color channel. Those should contain the segmentations (label images with suffix _labels.tif), as well as the intensity images. You will be asked to enter the names of the two color channel folders that you want to colocalize to obtain the properties of objects in the second channel within the ROI of the first channel. Please enter channel names in the corresponding order. Example: FITC DAPI would mean you want to obtain the regionprops of DAPI obects within FITC ROI. Then enter the suffix of the label images of each channel in the same order. Example: *_labels.tif *_labels.tif.'''))

        input_folder = popup_input("\nEnter the path to the parent folder: ")
        channels = input("\nEnter the names of the color channel subfolders in the abovementioned order (example: FITC DAPI): ")
        label_patterns = input("\nEnter the label patterns (example: *_labels.tif *_labels.tif): ")
        label_ids = input("\nEnter the label ids of the ROIs in the first channel (example: 1 2 3): ")
        ROI_size = input("\nDo you want to get the size of the ROI? (y/n): ")
        
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ROI_colocalization_regionprops.py',
                                        '--input ' + input_folder +
                                        ' --channels ' + channels +
                                        ' --label_patterns ' + label_patterns +
                                        ' --label_ids ' + label_ids +
                                        ' --ROI_size ' + ROI_size)
        restart_program()



    if choice == "6":
        os.system('clear')
        print("\n")
        print("----------------------------------------------------")
        print("You chose to get basic ROI properties (size, shape).")
        print("----------------------------------------------------")
        print("\n")
        print(wrapper.fill('''A popup will appear in a moment asking you to select the folder containing the label images.'''))

        input_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        label_pattern = input("\nEnter the label pattern of the label images (example: _labels.tif): ")
        channel = input("\nOf which channel do you want to quantify intensity? (1st: 0, 2nd: 1, 3rd: 2. Otherwise just press Enter): ")
        # allow no channel input
        channel = channel if channel else "-1"
        
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/get_basic_regionprops.py',
                                        '--input ' + input_folder +
                                        ' --label_pattern ' + label_pattern +
                                        ' --channel ' + channel)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()


def validation():
    os.system('clear')
    print("\nValidation: What would you like to do?\n")
    print("[1] Validate spot counts (2D)")
    print("[2] Validate blobs (2D or 3D; global F1 score)")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    

    if choice == "1":
        os.system('clear')
        print("\n")
        print("--------------------------------------------------------")
        print("You chose to validate spot counts against manual counts.")
        print("--------------------------------------------------------")
        print("\n")
        print(wrapper.fill('''A popup will appear in a moment asking you to select the folder containing the label images.'''))
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        label_pattern = input("\nEnter the label pattern of the label images (example: _labels.tif): ")
        gt_pattern = input("\nEnter the ground truth pattern of the label images (example: _ground_truth.tif): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/counts_validation.py',
                                        '--input ' + input_folder +
                                        ' --label_pattern ' + label_pattern +
                                        ' --gt_pattern ' + gt_pattern)
        restart_program()
    if choice == "2":
        os.system('clear')
        print("\n")
        print("---------------------------------------------------------------")
        print("You chose to validate blobs by calculating the global F1 score.")
        print("---------------------------------------------------------------")
        print("\n")
        print(wrapper.fill('''A popup will appear in a moment asking you to select the folder containing the label images.'''))
        print("\n")
        print(wrapper.fill("""The F1 score is the harmonic mean of precision and recall, and it ranges from 0 to 1. A high F1 score indicates that the prediction has both high precision (most blobs are correctly segmented) and high recall (most existing objects are detected)."""))
        input_folder = popup_input("\nEnter the path to the folder containing label images (automated segmentation + annotations): ")
        label_pattern = input("\nEnter the suffix of the label images from automated segmentation (example: _labels.tif): ")
        gt_pattern = input("\nEnter the suffix you chose for your annotated images (example: _ground_truth.tif): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_validation_f1_score.py',
                                        '--input ' + input_folder + 
                                        ' --label_pattern ' + label_pattern +
                                        ' --gt_pattern ' + gt_pattern)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()
 
def restart_program():
    print("\n")
    choice = input("\nYou are finished. Press Enter to restart the Image Analysis Suite or press x to exit.\n")
    if choice == "":
        os.system('clear')
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()


def exit_program():
    os.system('clear')
    print("\n")
    print("\n Okay, goodbye!\n")
    print("PS: If you need me later, just type 'assistance'.\n")
    exit()

# Start the program
welcome_message()