// Set batch mode to true to speed up the macro by disabling the display of images during processing.
setBatchMode(true);

// set directory to /mnt
File.setDefaultDir("/mnt");

// Define input and output directories. 
inputDir = getDirectory("Select input directory");
File.setDefaultDir(inputDir); // this is needed when the macro file is located elsewhere
outputDir = inputDir + File.separator + "extracted_TIFs" + File.separator ;
File.makeDirectory(outputDir);

// Get a list of all files in the input directory
fileList = getFileList(inputDir);

// Loop over all LIF files in the input directory
for (i = 0; i < fileList.length; i++) {
  // Check if the current file is a LIF file
  if (endsWith(fileList[i], ".lif")) {
    // Open the LIF file
    path = inputDir + fileList[i];
    //run("Bio-Formats Macro Extensions");
    // Loop over all series in the LIF file
    Ext.setId(inputDir+fileList[i]);
    Ext.getSeriesCount(seriesCount);
    sCount=seriesCount;
    for (s = 0; s < sCount; s++) {
      run("Bio-Formats Importer", "open=[" + path + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_"+(s));
      saveAs("Tiff",outputDir + fileList[i] + "_series" + s + ".tif");
      // Close the TIF stack
      close();
    }

  }
}

setBatchMode(false);
