// This macro rescales single-channel z-stack images in a directory and saves the resulting images in a subfolder named 'rescaled'.
// It prompts the user to input the pixel width and voxel depth in microns.
// It checks if the input images are in TIFF format and uses the Bio-Formats Importer plugin to open them.
// It also assumes that there are no other TIFF files in the input directory, besides the ones containing the z-stacks to be rescaled.

// Set batch mode to true to speed up the macro by disabling the display of images during processing.
setBatchMode(true);

// set directory to /mnt
File.setDefaultDir("/mnt");

inputDir = getDirectory("Select input directory");

// Create the output directory 'rescaled' in the input directory.
outputDir = inputDir + File.separator + "rescaled" + File.separator ;
File.makeDirectory(outputDir);

list = getFileList(inputDir);

// Prompt the user to input the pixel width and voxel depth in microns.
x_value = getNumber("Enter Pixel width (microns)", 1); y_value = x_value;
z_value = getNumber("Enter Voxel depth (microns)", 1);



for (i = 0; i < list.length; i++) {
   if (endsWith(list[i], ".tif")) {
      filePath = inputDir + list[i];
      run("Bio-Formats Importer", "open=" +inputDir + list[i] + " autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
      // Rescale the image using the specified pixel width and voxel depth.
      run("Scale...", "x="+ x_value + " y=" + y_value + " z=" + z_value + " interpolation=None");
      saveAs("Tiff", outputDir + list[i]);
      // Close the image to free up memory (even in batch mode).
      close("*"); 
   }
}

setBatchMode(false);
