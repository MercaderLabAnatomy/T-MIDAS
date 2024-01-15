// This macro splits multi-channel z-stack images into single-channel z-stack images.
// The resulting single-channel z-stacks are stored in a subfolder named 'single_channels'.
// It checks if the input images are in TIFF format and uses the Bio-Formats Importer plugin to open them.
// It also assumes that there are no other TIFF files in the input directory, besides the ones containing the z-stacks to be split.

// Set batch mode to true to speed up the macro by disabling the display of images during processing.
setBatchMode(true);

// Prompt user to enter a square number for the total number of tiles
n_total = -1;
while (n_total < 0 || Math.sqrt(n_total) % 1 != 0) {
  n_total = getNumber("Enter a square number for the expected number of tiles per image:", 16);
}

// Calculate the number of tiles per row/column
n = Math.sqrt(n_total);

// Define input and output directories. 
inputDir = getDirectory("Select input directory");
File.setDefaultDir(inputDir); // this is needed when the macro file is located elsewhere

// Get a list of all files in the input directory
list = getFileList(inputDir);

// Loop through each file
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
        if (c == 1) {
          channelName = "TRITC";
        } else if (c == 2) {
          channelName = "FITC";
        } else if (c == 3) {
          channelName = "DAPI";
        } else {
          print("Unknown channel");
        }
        outputDir = inputDir + File.separator + channelName + File.separator ;
        File.makeDirectory(outputDir);

        // Get image properties
        id = getImageID(); 
        title = getTitle(); 
        getLocationAndSize(locX, locY, sizeW, sizeH); 
        width = getWidth(); 
        height = getHeight(); 
        tileWidth = width / n; 
        tileHeight = height / n; 

        // Loop through each tile
        for (y = 0; y < n; y++) { 
          offsetY = y * height / n; 
          for (x = 0; x < n; x++) { 
            offsetX = x * width / n; 

            // Select the image, set location, duplicate, and crop
            selectImage(id); 
            call("ij.gui.ImageWindow.setNextLocation", locX + offsetX, locY + offsetY); 
            tileTitle = title + " [" + x + "," + y + "]"; 
            run("Duplicate...", "title=" + tileTitle); 
            makeRectangle(offsetX, offsetY, tileWidth, tileHeight); 
            run("Crop"); 

            // Save the current tile as a new TIF file in the output folder
            saveAs("Tiff", outputDir + "C" + c + "-" + list[i] + "_tile_" + x + "_" + y + ".tif"); 
          } 
        } 
      }  
      // Close the image to free up memory (even in batch mode).
      close("*"); 
   }
}

setBatchMode(false);
