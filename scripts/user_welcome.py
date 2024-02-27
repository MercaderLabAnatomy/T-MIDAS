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

def get_model_name():
    return subprocess.check_output("lscpu | grep -E 'Model name'", 
                                   shell=True).decode('utf-8').strip().replace("Model name:", "").replace(" ", "")

def get_hostname():
    return subprocess.check_output("hostname", 
                                   shell=True).decode('utf-8').strip() +" workstation"


def get_no_cores():
    return subprocess.check_output("lscpu | grep -E '^CPU\(s\):'", 
                                   shell=True).decode('utf-8').strip().replace(" ", "").replace("CPU(s):", "")

def get_model_name_GPU():
    model_name_GPU = subprocess.check_output("nvidia-smi --query-gpu=gpu_name --format=csv,noheader", 
                                             shell=True).decode('utf-8').strip()
    return model_name_GPU

def get_available_RAM_GPU():
    available_RAM_GPU = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", 
                                                shell=True).decode('utf-8').strip()
    return str(round(int(available_RAM_GPU) / 1024, 2)) + " GB"


def welcome_message():
    os.system('clear')
    print("""                                                                              
                                                      n
                                                     ñ░L
                   -≈r≈≈,.                .«╥g]]DÑÑRDMh╟╥,
                  ╘░░░░░░░░░░]n≈≈-,,╥╦%Ñ╠╦╦▄▄▄╫╫╫▓▓╫╫£░╟╩╫╫▓h,
                   V░╩╦╦╦╦╦░░≥╦╦╦╦▄▓▓▓▓▀▓▓▓▌▌▒╫▓▓▓▓▓ÑÑ╩b░╫╫▓⌐╫N
                    └Ü░░░░░û░░╦╦╗╫▓▓▀▀▀▀▒▒╫▄▓▓▓▓▓╫▒MÜ╦░░╫╫▓▌;▓Mµ
                      ╙╨░ñ░░░░░µ╦å╫▒▀▀▀╫╬░╩╫╫╫╬Ñ╨╜ñ░░░╦╫╬▓▌░╣▌╔▌
                      A╦╦NN╨Ü░░░µ═^`      ╓╦╩Ü░░░░░░╦╫╬▓▓▀╥▓▀╔▓▌
                      ░░░░░░░Ü╨`      .≤D╨░░░░░░░░╦╫▄▓▌▀╥▄▓▀╦▓▌╔
                      ░░░µ═^`      .gÑÜ░░░]░░░░╦╦╫╫▓▀░╦▓▓▀╠▓▓▀╥Å
                                 ┌0░░░░░░░░░░╦╫╫▓▀Ö╗╬▓▓▀╠▓▓▌Ü╬▓
"""+str(get_hostname())+"""       ╓Ñ░░░░░░░╨╩╬╦╫Ñ╨░╗╫▓▓▀╠╬▓▓▀Ü╦╬▌MÜ`
   Image Analysis Suite      ¿Ñ░░░░░b╦░░░░░╫Å╬╫╫╫╣╫Ü╦╣▌▀░╬╫╫╩*`
                            ê░]░░╦╩░▄▄╙╫░░░╫╫╫╫╬╬╫╣╫▀Ü╗╬╝"
                           ╬╦╦░░░╫H░▀▀░╫╫╫╫╫╫ÑBÑ╨Ñ╦Ñ╨^
                           ╙╬╫╫╫╫╫╫RN╦╫╫╫╫Ñ╫░BÑ╨^        Maintenance:
                              *╩Ñ░╨╫ÑÑ╠Ñ╩╨`           marco.meer@unibe.ch
""")
    available_RAM = get_available_RAM()
    model_name = get_model_name()
    no_cores = get_no_cores()
    model_name_GPU = get_model_name_GPU()
    available_RAM_GPU = get_available_RAM_GPU()

    print(f"Available RAM: {available_RAM}")
    print(f"CPU Model: {model_name}")
    print(f"CPU cores: {no_cores}")
    print(f"GPU Model: {model_name_GPU}")
    print(f"Available GPU Memory: {available_RAM_GPU}")
    
    global user_name
    if 'user_name' not in globals():
        user_name = input("\nWhat's your name? ")

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

    print(f"\nHi {user_name}! What would you like to do?\n")
    print("[1] Image Preprocessing")
    print("[2] Image Segmentation or Spot Detection")
    print("[3] Regions of Interest Analysis")
    print("[4] Validation")
    print("[5] Special Workflows")
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
    if choice == "5":
        workflows()
        restart_program()
    if choice == "x" or choice == "X":
        exit_program()
    

def image_preprocessing():
    
    os.system('clear')
    print("Image Preprocessing: What would you like to do?\n")
    print("[1] File Conversion")
    print("[2] Image Cropping")
    print("[3] Split Channels")
    print("[4] Maximum Intensity Projection (MIP)")
    print("[5] Image Tiling")
    print("[6] Sample Random Tiles")
    print("[7] Image Denoising")
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
        print("\nSplit Channels: Opening Fiji macro...")
        subprocess.Popen("/opt/Image_Analysis_Suite/macros/batch_channel_splitter.ijm".split(), 
                         stdout=subprocess.PIPE)
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
        #input("\nTakes images with dimension order XY or CXY (Fiji ). Press Enter to continue...")
        input_folder = popup_input("\nEnter the path to the folder containing the (multichannel) .tif images: ")
        tile_diagonal = input("\nEnter the tile diagonal in pixels: ")
        percentage = input("\nEnter the percentage of random tiles to be picked from the entire image (20-100): ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/random_tile_sampler.py',
                                        '--input ' + input_folder + ' --tile_diagonal ' + tile_diagonal + ' --percentage ' + percentage)
        restart_program()
    if choice == "7":
        Image_Denoising()
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()


def Image_Denoising():
    os.system('clear')
    print("\nImage Denoising: What would you like to do?\n")
    print("[1] Denoise .tif images")
    print("[2] Denoise .ser images from TEM")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    if choice == "1":
        print("\nYou chose to denoise images. Opening Aydin Studio...")
        
        subprocess.Popen("mamba run -n aydin_env aydin".split(), stdout=subprocess.PIPE, cwd="/mnt/")
        restart_program()
    if choice == "2":
        print("\nYou chose to denoise .ser images from TEM. Running Python script...")
        input_folder = popup_input("\nEnter the path to the folder containing the .ser images: ")
        python_script_environment_setup('em_denoising', 
                                        '/opt/Image_Analysis_Suite/scripts/denoise_em.py',
                                        '--input_folder ' + input_folder)
        print('''
              \nTo see the resolution in Fiji / ImageJ, 
              .tif files must be imported using Files > Import > Bio-Formats.
              ''')
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
    print("[3] Convert .czi")
    print("[4] Convert brightfield .czi")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    if choice == "1":
        os.system('clear')
        print("\nFile Conversion (.ndpi): Running python script...)")
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi(s) files: ")
        # ask for resolution level of the ndpi image
        LEVEL = input('''
                      \nEnter the desired resolution level 
                      (0 = highest resolution, 1 = second highest resolution): 
                      ''')
        print('''\nConverting .ndpi files to .tif files. 
              Beware: If the image is too large this will raise an exception. 
              In that case, better use ndpi file cropping.
              ''')
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/ndpi2tif.py',
                                        '--input ' + input_folder + ' --level ' + LEVEL)
        restart_program()
    if choice == "2":
        os.system('clear')
        print("\nFile Conversion (.lif): Opening Fiji macro...")
        subprocess.Popen("/opt/fiji/ImageJ-linux64 -macro /opt/macros/LIFs2TIFs.ijm".split(), 
                         stdout=subprocess.PIPE)
        restart_program()
    if choice == "3":
        os.system('clear')
        print("\nFile Conversion (.czi): Opening Fiji macro...")
        subprocess.Popen("/opt/fiji/ImageJ-linux64 -macro /opt/macros/batch_CZI_to_8bitTIF.ijm".split(), 
                         stdout=subprocess.PIPE)
        restart_program()
    if choice == "4":
        os.system('clear')
        print("\nFile Conversion (brightfield .czi):")
        input_folder = popup_input("\nEnter the path to the folder containing the brightfield .czi(s) files: ")
        scale_factor = input("\nEnter the scale factor (0.5 = half the size (default)): ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/brightfield_czi_to_tif.py',
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
    print("[1] Manually (2D)")
    print("[2] Automatically (currently only for heart slices (.ndpi) and hearts (.lif))")
    print("[3] Automatically (brightfield .ndpi)")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")

    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    if choice == "1":
        print('''
              \nYou chose manual. Napari will open in a moment. 
              \n Choose Tools > Utilities > Crop Regions for 2D
              \n It prefers images in TIF format. 
              \nFor further info, see https://github.com/biapol/napari-crop
              ''')
        napari_environment_setup('napari-assistant')
        restart_program()
    if choice == "2":
        print("\nYou chose automatic.")
        print("\nWhich file format?")
        print("\n1. .ndpi")
        print("\n2 .lif")        
        sub_choice = input("\nEnter your choice: ")
        
        if sub_choice == "1":
            input_folder = popup_input("\nEnter the path to the folder containing ndpi(s) files: ")
            CROPPING_TEMPLATE_CHANNEL_NAME = input('''
                                                   Enter the channel name 
                                                   that represents the cropping template 
                                                   (for example FITC or CY5): 
                                                   ''')
            python_script_environment_setup('napari-assistant', 
                                            '/opt/Image_Analysis_Suite/scripts/ndpi2croppedtif.py',
                                            '--input ' + input_folder + 
                                            ' --cropping_template_channel_name ' + CROPPING_TEMPLATE_CHANNEL_NAME)
            restart_program()            
        if sub_choice == "2":
            print("\nYou chose to crop .lif files.")
            lif = popup_input("\nEnter the path to the .lif file: ")
            template_channel = input('''
                                     Enter the channel number 
                                     that represents the cropping template 
                                     (single channel: 0): 
                                     ''')
            python_script_environment_setup('napari-assistant', 
                                            '/opt/Image_Analysis_Suite/scripts/lif_to_cropped_tifs.py',
                                            '--lif ' + lif + ' --template_channel ' + template_channel)
            restart_program()
        else:
            print("Invalid choice")
            restart_program() 
    if choice == "3":
        print("\nYou chose to crop brightfield .ndpi files.")
        input_folder = popup_input("\nEnter the path to the folder containing the .ndpi files: ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/brightfield_ndpi2croppedtif.py',
                                        '--input ' + input_folder)
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
    print("\nImage Segmentation or Spot Detection: What would you like to do?\n")
    print("[1] Manually segment objects (2D) with Napari McLabel plugin")
    print("[2] Train pixel classifier to segment objects (2D or 3D) with Napari APOC plugin")
    print("[3] Automatically segment bright spots (2D)")
    print("[4] Automatically segment nuclei (3D; requires dark background and good SNR)")
    print("[5] Automatically segment tissue (3D; requires dark background and good SNR)")
    print("[6] Automatically segment myocardium in cropped heart slices (2D)")
    print("[r] Return to Main Menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])
    if choice == "1":
        print("\nYou chose to manually segment objects in 2D with McLabel. Napari will open in just a moment.")
        napari_environment_setup("napari-assistant")
        restart_program()  
    if choice == "2":
        print("\nYou chose to train a pixel classifier to segment objects (2D or 3D). Napari will open in just a moment.")
        napari_environment_setup("napari-assistant")
        restart_program()       
    if choice == "3":
        print("\nYou chose to automatically segment bright spots in 2D.")
        input_folder = popup_input("\nEnter the path to the folder containing the intensity images: ")
        bg = input("\nWhat kind of background? (1 = gray, 2 = dark): ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/2D_segmentation_spots.py',
                                        '--input ' + input_folder + ' --bg ' + bg)
        restart_program()
    if choice == "4":
        print("\nYou chose to automatically segment nuclei in 3D.")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        nuclei_channel = input("\nEnter the channel number that represents the nuclei channel: ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_segment_nuclei.py',
                                        '--image_folder ' + input_folder + ' --nuclei_channel ' + nuclei_channel)
        restart_program()
        sub_choice = input("\nEnter your choice: ")
    if choice == "5":
        print("\nYou chose to automatically segment tissue in 3D.")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        tissue_channel = input("\nEnter the channel number that represents the tissue channel: ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_segment_tissue.py',
                                        '--image_folder ' + input_folder + ' --tissue_channel ' + tissue_channel)
        restart_program()
    if choice == "6":
        print("\nYou chose to automatically segment myocardium in cropped heart slices (2D).")
        input_folder = popup_input("\nEnter the path to the folder containing the .tif images: ")
        image_type = input("\nBrightfield images? (y/n): ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/get_myocardium_from_slices.py',
                                        '--input ' + input_folder + ' --image_type ' + image_type)
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
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/create_ventricle_ROIs.py',
                                        '--input ' + input_folder + ' --pixel_resolution ' + pixel_resolution)
        restart_program()
    if choice == "2":
        print("\nYou chose to count spots in 2D ROIs within ventricle slices.")
        pixel_resolution = input("\nEnter the pixel resolution of the images in um/px: ")
        input_folder = popup_input("\nInput: Folder with all label images (ROIs and instance segmentations).")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/count_instances_per_ROI.py',
                                        '--input ' + input_folder + ' --pixel_resolution ' + pixel_resolution)
        restart_program() 
    if choice == "3":
        print("\nYou chose to count nuclei in tissue (3D).")
        nuclei_folder = popup_input("\nEnter the path to the folder containing nuclei label images: ")
        tissue_folder = popup_input("\nEnter the path to the folder containing tissue label images: ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_count_nuclei_in_tissue.py',
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
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/counts_validation.py',
                                        '--input ' + input_folder)
        restart_program()
    if choice == "2":
        print("\nYou chose to validate segmentation results against manual segmentation results.")
        print("\nNames of your manually annotated label images must end with '_ground_truth.tif'.")
        input_folder = popup_input("\nEnter the path to the folder containing the segmentation results: ")
        segmentation_type = input("\nHow many labels do the label images contain? (s = single, m = multiple) ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/3D_nuclei_validation.py',
                                        '--input ' + input_folder + ' --type ' + segmentation_type)
        restart_program()
    if choice == "r" or choice == "R":
        welcome_message()
    if choice == "x" or choice == "X":
        exit_program()
    else:
        print("Invalid choice")
        restart_program()
 

def workflows():
    os.system('clear')
    print("\nSpecial Workflows: Choose from one of the available workflows\n")
    print("[1] Analyze multichannel .tif (ndpi viewer exports) of CM culture wells")
    print("[2] Count proliferating FITC+ cells")
    print("[r] Return to main menu")
    print("[x] Exit \n")
    choice = input("\nEnter your choice: ")
    user_choices.append([choice])

    if choice == "1":
        print("\nYou chose to analyze multichannel .tif of CM culture wells.")
        input_folder = popup_input("\nEnter the path to the folder containing the multichannel .tif files: ")
        tile_diagonal = input("\nEnter the tile diagonal in pixels: ")
        # num_tiles = input("\nEnter the number of tiles to sample: ")
        channel_names = input('''
                              \nEnter the names of the channels 
                              in the order they appear 
                              in the .tif files. 
                              Example: DAPI GFP RFP: 
                              ''')
        python_script_environment_setup('workflow_CM_culture', 
                                        '/opt/Image_Analysis_Suite/scripts/analyze_cm_culture_wells.py',
                                        '--input ' + input_folder + ' --tile_diagonal ' + tile_diagonal + 
                                        ' --channels ' + channel_names)
        restart_program()
    if choice == "2":
        print("\nYou chose to analyze the output of the previous workflow step.")
        input_folder = popup_input("\nEnter the path to the folder containing the channel folders: ")
        python_script_environment_setup('napari-assistant', 
                                        '/opt/Image_Analysis_Suite/scripts/regionprops_CM_culture.py',
                                        '--input ' + input_folder)
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
