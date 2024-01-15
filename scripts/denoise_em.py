import os
import numpy as np
from tk_r_em import load_network
import ncempy.io as nio
import tensorflow as tf
from pyometiff import OMETIFFWriter
import argparse

tf.config.run_functions_eagerly(True)



# get input folder from command line
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_folder', metavar='input_folder', type=str,
                    help='path to the folder containing .ser files to be denoised')
args = parser.parse_args()


def fcn_set_gpu_id(gpu_visible_devices: str = "0") -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visible_devices

fcn_set_gpu_id("0")

# check if gpu is available
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#ser_file = '/mnt/disk1/Marco/Carol_Muscle_Cox7a1-_WT/SkeletalMuscle_7_cox7a1KO/Sample7-1c1_200nm_1.ser'


def load_ser_file(ser_file):
    # read .ser file
    ser_data = nio.ser.serReader(ser_file)
    if "metadata" not in ser_data:
        logger.warning(f"No metadata found for file {ser_file}. Returning empty dict.")
        return {}, ser_data["data"], ser_data["pixelSize"]
    return ser_data["metadata"], ser_data["data"], ser_data["pixelSize"]

#test = load_ser_file(ser_file)


# check if test[1] is a numpy array
#isinstance(test[1], np.ndarray)


def fcn_inference(x, net_name):
    """
    Perform inference on test data using a pre-trained model.
    Args:
        x (numpy.ndarray): Input data.
        net_name (str): Name of the network.
    Returns:
        y_p (numpy.ndarray): Predicted output.
    """
    r_em_nn = load_network(net_name)
    r_em_nn.summary()

    #n_data = x.shape[0]
    #batch_size = 8

    # run inference
    #y_p = r_em_nn.predict(x, batch_size)
    
    # weak GPU
    y_p = r_em_nn.predict_patch_based(x, patch_size=128, stride=128, batch_size=8)

    return y_p

def process_ser_folder(input_folder, output_folder):
    """
    Process a folder containing TIF files, perform inference, and save the output as 16-bit TIF files.
    
    Args:
        input_folder (str): Path to the folder containing input TIF files.
        output_folder (str): Path to the folder where output TIF files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the input folder
    ser_files = [f for f in os.listdir(input_folder) if f.endswith('.ser')]
    
    for ser_file in ser_files:
        # Load the input TIF file
        input_path = os.path.join(input_folder, ser_file)
        # read 16bit grayscale image using nio
        metadata, input_image, pixel_size = load_ser_file(input_path)
        #input_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)
        #print(pixel_size)
        #print(1/ (pixel_size[0] *1e13))
        # # Assuming you have a function 'process_image' for inference
        net_name = 'sfr_lrtem'  # Low-Resolution Transmission Electron Microscopy 
        output_image = fcn_inference(input_image, net_name)
        # # convert to uint16
        # output_image = output_image.astype(np.uint16)
        
        #output_image = input_image
        # convert output image to a 3D numpy array
        output_array = np.expand_dims(output_image, axis=0)
        # Save the output as 16-bit OME-TIF using input_image and pixel_size
        output_path = os.path.join(output_folder, ser_file)
        # change file ending to .tif
        output_path = output_path[:-4] + '.tif'
                        
        metadata_dict = {
            # pixel size in µm is 
            "PhysicalSizeX": 1/ (pixel_size[0] *1e10 * 1000),
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 1/ (pixel_size[1] *1e10 * 1000),
            "PhysicalSizeYUnit": "µm",
            "SizeX": output_array.shape[1],
            "SizeY": output_array.shape[0],

 
        }
            
        writer = OMETIFFWriter(fpath=output_path, array=output_array, metadata=metadata_dict, dimension_order='ZYX',explicit_tiffdata=False)
        writer.write()
        

if __name__ == '__main__':
    input_folder = args.input_folder
    #input("Enter the path to the folder containing .ser files to be denoised: ")
    output_folder = os.path.join(input_folder, "denoised")
    
    process_ser_folder(input_folder, output_folder)
    print('Processing complete.')
