import subprocess
import os 
import tkinter as tk
from tkinter import filedialog
import csv
import datetime
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d %H:%M:%S")

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

def napari_environment_setup(env_name):
    subprocess.run(f"mamba run -n {env_name} napari".split(),cwd="/mnt/")
    restart_program()


def python_script_environment_setup(env_name, script_name, input_parameters=None):
    subprocess.run(f"mamba run -n {env_name} python {script_name} {input_parameters}".split(),
                   capture_output=False,text=True,cwd="/mnt/")
    restart_program()

def popup_input(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = filedialog.askdirectory(title=prompt, initialdir="/mnt/")
    return user_input


user_choices = []

def main_menu():
    os.system('clear')
    print(f"\nHi {user_name}! What would you like to do?\n")
    print("[1] Image Preprocessing")
    print("[2] Image Segmentation")
    print("[3] Regions of Interest (ROI) Analysis")
    print("[4] Image Segmentation Validation")
    print("[x] Exit \n")
    
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    
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
    print("[1] File Conversion")
    print("[2] Image Cropping")
    print("[3] Split Color Channels")
    print("[4] Maximum Intensity Projection (MIP)")
    print("[5] Image Tiling")
    print("[6] Sample Random Tiles")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")

    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    
    if choice == "1":
        file_conversion()
        restart_program()
    if choice == "2":
        crop_images()
        restart_program()
    if choice == "3":
        os.system('clear')
        print("\nSplit Channels: ")
        input_folder = popup_input("\nEnter the path to the folder containing the multichannel .tif images: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/split_color_channels.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "4":
        os.system('clear')       
        print("\nMIP: Opening Fiji macro...")
        subprocess.Popen("/opt/Image_Analysis_Suite/macros/batch_MIP.ijm".split(), 
                         stdout=subprocess.PIPE)
        restart_program()
    if choice == "5":
        os.system('clear')
        print("\nImage Tiling: Opening Fiji macro...")
        subprocess.Popen("/opt/Image_Analysis_Suite/macros/batch_tile_2D_images.ijm".split(), 
                         stdout=subprocess.PIPE)
        restart_program()
    if choice == "6":
        os.system('clear')
        print('''
              \nRandom Tile Sampling: If your image consists of multiple channels, 
              please provide all channels in a single multichannel .tif image.
              ''')
        input_folder = popup_input("\nEnter the path to the folder containing the (multichannel) .tif images: ")
        tile_diagonal = input("\nEnter the tile diagonal in pixels: ")
        percentage = input("\nEnter the percentage of random tiles to be picked from the entire image (20-100): ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/random_tile_sampler.py',
                                        '--input ' + input_folder + ' --tile_diagonal ' + tile_diagonal + ' --percentage ' + percentage)
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
    print("\nFile Conversion: Which file format would you like to convert?\n")
    print("[1] Convert .ndpi")
    print("[2] Convert .lif")
    print("[3] Convert brightfield .czi")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    if choice == "1":
        os.system('clear')
        print("\nFile Conversion (.ndpi): \n A popup will appear in a moment asking you to select the folder containing the .lif files.")
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
                                        '/opt/Image_Analysis_Suite/scripts/ndpis_to_tifs.py',
                                        '--input ' + input_folder + ' --level ' + LEVEL)
        restart_program()
    if choice == "2":
        os.system('clear')
        print('''File Conversion (.lif): 
              \n A popup will appear in a moment asking you to select the folder containing the .lif files. 
              Scenes of each .lif will be exported as .tif files with resolution metadata.''')
        input_folder = popup_input("\nEnter the path to the folder containing the .lif file: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/lif_to_tifs.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "3":
        os.system('clear')
        print("\nFile Conversion (brightfield .czi):")
        input_folder = popup_input("\nEnter the path to the folder containing the brightfield .czi(s) files: ")
        scale_factor = input("\nEnter the scale factor (0.5 = half the size (default)): ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/czi_to_tif_brightfield.py',
                                        '--input ' + input_folder + 'scale_factor ' + scale_factor)
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
    user_choices.append([choice])
    if choice == "1":
        os.system('clear')
        input_folder = popup_input("\nEnter the path to the folder containing ndpi(s) files: ")
        CROPPING_TEMPLATE_CHANNEL_NAME = input('''
                                               Enter the channel name 
                                               that represents the cropping template 
                                               (for example FITC or CY5): 
                                               ''')
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/ndpis_to_cropped_tifs.py',
                                        '--input ' + input_folder + 
                                        ' --cropping_template_channel_name ' + CROPPING_TEMPLATE_CHANNEL_NAME)
        restart_program()            

    if choice == "2":
        print("\nYou chose to crop brightfield .ndpi files.")
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi files: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/ndpis_to_cropped_tifs_brightfield.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "3":
        os.system('clear')
        print("\nYou chose to crop .lif files.")
        input_folder = popup_input("\nEnter the path to the .lif file: ")
        template_channel = input('''
                                 Enter the channel number 
                                 that represents the cropping template 
                                 (single channel: 0): 
                                 ''')
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/lif_to_cropped_tifs.py',
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
    print("[2] Segment blobs (3D; requires dark background and good SNR)")
    print("[3] Semantic segmentation (2D, fluorescence or brightfield)")
    print("[4] Semantic segmentation (3D; requires dark background and good SNR)")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
   
    if choice == "1":
        print("\nYou chose to automatically segment bright spots in 2D.")
        input_folder = popup_input("\nEnter the path to the folder containing the intensity images: ")
        bg = input("\nWhat kind of background? (1 = gray, 2 = dark): ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/2D_segmentation_spots.py',
                                        '--input ' + input_folder + ' --bg ' + bg)
        restart_program()
    if choice == "2":
        print("\nYou chose to automatically segment blobs in 3D.")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        nuclei_channel = input("\nEnter number of the color channel you want to segment:  ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_segment_instances.py',
                                        '--image_folder ' + input_folder + ' --nuclei_channel ' + nuclei_channel)
        restart_program()
    if choice == "3":
        print("\nYou chose semantic segmentation (2D, fluorescence or brightfield).")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        image_type = input("\nBrightfield images? (y/n): ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/get_myocardium_from_slices.py',
                                        '--input ' + input_folder + ' --image_type ' + image_type)
        restart_program()
    if choice == "4":
        print("\nYou chose semantic segmentation (3D).")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        tissue_channel = input("\nEnter number of the color channel you want to segment: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_segment_semantic.py',
                                        '--image_folder ' + input_folder + ' --tissue_channel ' + tissue_channel)
        restart_program()

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
    print("[1] Heart slices: Generate ROIs from [intact+injured] ventricle masks")
    print("[2] Heart slices: Count spots within ventricle ROIs")
    print("[3] Heart volume: Count nuclei within ROIs")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])

    if choice == "1":
        print("\nYou chose to create ROIs from masks.")
        pixel_resolution = input("\nEnter the pixel resolution of the images in um/px: ")
        input_folder = popup_input("\nEnter the path to the folder containing the label images: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/create_ventricle_ROIs.py',
                                        '--input ' + input_folder + ' --pixel_resolution ' + pixel_resolution)
        restart_program()
    if choice == "2":
        print("\nYou chose to count spots in 2D ROIs within ventricle slices.")
        pixel_resolution = input("\nEnter the pixel resolution of the images in um/px: ")
        input_folder = popup_input("\nInput: Folder with all label images (ROIs and instance segmentations).")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/count_instances_per_ROI.py',
                                        '--input ' + input_folder + ' --pixel_resolution ' + pixel_resolution)
        restart_program() 
    if choice == "3":
        print("\nYou chose to count nuclei in tissue (3D).")
        nuclei_folder = popup_input("\nEnter the path to the folder containing nuclei label images: ")
        tissue_folder = popup_input("\nEnter the path to the folder containing tissue label images: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_count_instances_in_ROIs.py',
                                        '--nuclei_folder ' + nuclei_folder + ' --tissue_folder ' + tissue_folder)
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
    print("[1] Validate predicted counts against manual counts (2D label images)")
    print("[2] Validate predicted segmentation results against manual segmentation results (2D or 3D label images)")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])

    if choice == "1":
        print("\nYou chose to validate predicted counts against manual counts.")
        print("\nNames of your manually annotated label images must end with '_ground_truth.tif'.")
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/counts_validation.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "2":
        print("\nYou chose to validate segmentation results against manual segmentation results.")
        print("\nNames of your manually annotated label images must end with '_ground_truth.tif'.")
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        segmentation_type = input("\nHow many labels do the label images contain? (s = single, m = multiple) ")
        python_script_environment_setup('tmidas-env', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_segment_instances_validation.py',
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
    choice = input("\n When you are finished, press Enter to restart the Image Analysis Suite or press x to exit.\n")
    if choice == "":
        os.system('clear')
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()


def exit_program():
    os.system('clear')
    print("\n Okay, goodbye!\n")
    print("PS: If you need me later, just type 'assistance'.\n")
    
    # print log
    print(f"On {date}, {user_name} chose the following workflow: {', '.join(map(str, user_choices))}.")
    # add to log.txt
    with open('/opt/Image_Analysis_Suite/log.txt', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([date, user_name, user_choices])

    exit()

# Start the program
welcome_message()