// Set batch mode to true to speed up the macro by disabling the display of images during processing.
setBatchMode(true);


// set directory to /mnt
File.setDefaultDir("/mnt");

// Define input and output directories. 
inputDir = getDirectory("Select input directory");
File.setDefaultDir(inputDir); // this is needed when the macro file is located elsewhere
outputDir = inputDir + File.separator + "max_intensity_projections" + File.separator ;
File.makeDirectory(outputDir);

// Get a list of all files in the input directory
list = getFileList(inputDir);

for (i = 0; i < list.length; i++) {
  if (endsWith(list[i], ".tif")) {
    // Open the z-stack image using Bio-Formats Importer plugin
    run("Bio-Formats Importer", "open=[" + inputDir + list[i] + "] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
    // get some info about the image
    getDimensions(width, height, channels, slices, frames);
    // Check if the image is a z-stack (i.e., has more than one slice)
    if (slices > 1) {
      run("Z Project...", "projection=[Max Intensity]");
      // Set up output file name
      outputFile = outputDir + File.separator + replace(list[i], ".tif", "_mip.tif");
      saveAs("Tiff", outputFile);
      // Close the image to free up memory
      close("*");
    }
  }
}

setBatchMode(false);
