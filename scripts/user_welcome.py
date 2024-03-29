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
    print(f"\nRunning chosen pipeline with the following parameters: {input_parameters}")
    subprocess.run(f"mamba run -n {env_name} python {script_name} {input_parameters}".split(),
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
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()
    

def image_preprocessing():
    
    os.system('clear')
    print("Image Preprocessing: What would you like to do?\n")
    print("[1] File Conversion to TIFF")
    print("[2] Cropping Largest Objects from Images")
    print("[3] Extract intersecting regions of two images")
    print("[4] Sample Random Image Subregions")
    print("[5] Normalize intensity across single color image (CLAHE)")
    print("[6] Split color channels of multicolor images")
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
        print(wrapper.fill("You chose to split the color channels of multicolor images. A popup will appear in a moment asking you to select the folder containing the multicolor images. You will be asked to enter the dimension order of the images and the names of the color channel output folders."))
        print("\n")
        input_folder = popup_input("\nEnter the path to the folder containing the multicolor images: ")
        dim_order = input("\nEnter the dimension order of the images (example: XYZCT): ")
        channel_names = input("\nEnter the names of the color channels (example: FITC DAPI TRITC): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/split_color_channels.py',
                                        '--input ' + input_folder + ' --dim_order ' + dim_order + ' --channel_names ' + channel_names)
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
    print("[2] Convert .lif")
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
        print('''You chose to convert .lif files to .tif files.\n
              A popup will appear in a moment asking you to select the folder containing the .lif files. 
              Scenes of each .lif will be exported as .tif files with resolution metadata.''')
        input_folder = popup_input("\nEnter the path to the folder containing the .lif file: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/lif_to_tifs.py',
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
        CROPPING_TEMPLATE_CHANNEL_NAME = input('''
                                               Enter the channel name 
                                               that represents the cropping template 
                                               (for example FITC or CY5): 
                                               ''')
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ndpis_to_cropped_tifs.py',
                                        '--input ' + input_folder + 
                                        ' --cropping_template_channel_name ' + CROPPING_TEMPLATE_CHANNEL_NAME)
        restart_program()            

    if choice == "2":
        print('''You chose to crop blobs from brightfield .ndpi files. \n
              A popup will appear in a moment asking you to select the folder containing ndpi files.''')
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi files: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/ndpis_to_cropped_tifs_brightfield.py',
                                        '--input ' + input_folder)
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
    print("[1] Segment bright spots (2D)")
    print("[2] Segment blobs (2D)")
    print("[3] Segment blobs (3D; requires dark background and good SNR)")
    print("[4] Semantic segmentation (2D, fluorescence or brightfield)")
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
        bg = input("\nWhat kind of background? (1 = gray, 2 = dark): ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_spots_2D.py',
                                        '--input ' + input_folder + ' --bg ' + bg)
        restart_program()
        
    if choice == "2":
        os.system('clear')
        print('''You chose to segment blobs in 2D. \n
                A popup will appear in a moment asking you to select the folder containing the single color .tif images.
                ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        sigma = input("\nEnter the sigma for the gauss-otsu-labeling (example: 1.0): ")
        exclude_small = input("\nLower size threshold to exclude small objects: ")
        exclude_large = input("\nUpper size threshold to exclude large objects: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_blobs_2D.py',
                                        '--input ' + input_folder + 
                                        ' --sigma ' + sigma +
                                        ' --exclude_small ' + exclude_small + 
                                        ' --exclude_large ' + exclude_large)
        restart_program()
        
    
    if choice == "3":
        os.system('clear')
        print('''You chose to segment blobs in 3D. \n
              A popup will appear in a moment asking you to select the folder containing the .tif images.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        nuclei_channel = input("\nEnter number of the color channel you want to segment:  ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_instances_3D.py',
                                        '--image_folder ' + input_folder + ' --nuclei_channel ' + nuclei_channel)
        restart_program()
    if choice == "4":
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
    if choice == "5":
        os.system('clear')
        print('''You chose semantic segmentation (3D). \n
                A popup will appear in a moment asking you to select the folder containing the .tif images.
                ''')
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        tissue_channel = input("\nEnter number of the color channel you want to segment: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_semantic_3D.py',
                                        '--image_folder ' + input_folder + ' --tissue_channel ' + tissue_channel)
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
    print("[1] Heart slices: Generate ROI from [intact+injured] ventricle masks")
    print("[2] Count spots within ROI (2D)")
    print("[3] Count blobs within ROI (3D)")
    print("[4] Colocalize ROI in different color channels")
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
        print(wrapper.fill("""Input data structure: A popup will appear in a moment asking you to select the parent folder containing a subfolder for each color channel. Those should contain the segmentations (label images with suffix _labels.tif). You will be asked to enter the names of all color channel folders. Please enter them in the order in which you want to colocalize them. Example: FITC DAPI TRITC would mean you want to count DAPI in FITC and TRITC in DAPI and FITC. Then enter the suffix of the label images of each channel in the same order. Example: *_labels.tif *_labels.tif *_labels.tif. 
                           """))
        print("\n")

        input_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        channels = input("\nEnter the names of your color channel subfolders in the abovementioned order (example: FITC DAPI TRITC): ")
        label_patterns = input("\nEnter the label patterns (example: *_labels.tif *_labels.tif *_labels.tif): ")
        #add_intensity = input("\nDo you want to quantify average intensity of the last channel in the ROI of the second last channel? (y/n): ")
        output_images = input("\nDo you want to save colocalization images? (y/n): ")
        get_area = input("\nDo you want to get the area of the ROI of the first channel? (y/n): ")
        python_script_environment_setup('tmidas-env', 
                                    os.environ.get("TMIDAS_PATH")+'/scripts/ROI_colocalization_multicolor.py',
                                    '--input ' + input_folder +
                                    ' --channels ' + channels +
                                    ' --label_patterns ' + label_patterns +
                                    ' --output_images ' + output_images +
                                    ' --get_area ' + get_area
                                    )
                                    #' --add_intensity ' + add_intensity
                                    
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
    print("[1] Validate counts against manual counts (2D label images)")
    print("[2] Validate segmentation results against manual segmentation results (2D or 3D label images)")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    

    if choice == "1":
        os.system('clear')
        print('''You chose to validate predicted counts against manual counts. \n
                A popup will appear in a moment asking you to select the folder containing the segmentation results.'''
        )
        print("\nNames of your manually annotated label images must end with '_ground_truth.tif'.")
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/counts_validation.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "2":
        os.system('clear')
        print('''You chose to validate segmentation results against manual segmentation results. \n
                A popup will appear in a moment asking you to select the folder containing the segmentation results.'''
        )
        print("\nNames of your manually annotated label images must end with '_ground_truth.tif'.")
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        segmentation_type = input("\nHow many labels do the label images contain? (s = single, m = multiple) ")
        python_script_environment_setup('tmidas-env', 
                                        os.environ.get("TMIDAS_PATH")+'/scripts/segmentation_instances_validation.py',
                                        '--input ' + input_folder + ' --type ' + segmentation_type)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()
 
def restart_program():
    choice = input("\n You are finished. Press Enter to restart the Image Analysis Suite or press x to exit.\n")
    if choice == "":
        os.system('clear')
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()


def exit_program():
    os.system('clear')
    print("\n Okay, goodbye!\n")
    print("PS: If you need me later, just type 'assistance'.\n")
    exit()

# Start the program
welcome_message()