// This macro splits multi-channel z-stack images into single-channel z-stack images.
// The resulting single-channel z-stacks are stored in a subfolder named 'single_channels'.
// It checks if the input images are in TIFF format and uses the Bio-Formats Importer plugin to open them.
// It also assumes that there are no other TIFF files in the input directory, besides the ones containing the z-stacks to be split.

// Set batch mode to true to speed up the macro by disabling the display of images during processing.
setBatchMode(true);

// set directory to /mnt
File.setDefaultDir("/mnt");

// Define input and output directories. 
inputDir = getDirectory("Select input directory");
File.setDefaultDir(inputDir); // this is needed when the macro file is located elsewhere
outputDir = inputDir + File.separator + "single_channels" + File.separator ;
File.makeDirectory(outputDir);

// Get a list of all files in the input directory
list = getFileList(inputDir);


for (i = 0; i < list.length; i++) {
   if (endsWith(list[i], ".tif")) {
      filePath = inputDir + list[i];
      run("Bio-Formats Importer", "open=[" +inputDir + list[i] + "] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
      // get some info about the image
	  getDimensions(width, height, channels, slices, frames);
	  if (channels > 1) run("Split Channels");
      // Get the number of resulting images after splitting.
      nChannels = nImages();
      for (c = 1; c <= nChannels; c++) {
      selectImage(c);
      // Get the name of the current channel
      channelName = getTitle();
      saveAs("Tiff", outputDir + channelName );
      }  
      // // Close the image to free up memory (even in batch mode).
      close("*"); 
   }
}

setBatchMode(false);
