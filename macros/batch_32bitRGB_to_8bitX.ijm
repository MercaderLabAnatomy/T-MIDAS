// This script converts 32-bit RGB TIFF to 8-bit single color TIFF by splitting color channels and saving the selected color channel

// Enable batch mode to speed up the macro by disabling the display of images during processing.
setBatchMode(true);

// set directory to /mnt
File.setDefaultDir("/mnt");

// Prompt the user to select the directory containing the images
inputFolder = getDirectory("Select input directory");

// Get a list of TIFF files in the folder
list = getFileList(inputFolder);

// Set the bit depth to 8
bit_depth = 8;

// Prompt the user to enter a letter representing the color channel (R, G, or B)
inputString = getString("Enter a color channel (R, G, or B):", "B");

// Convert the input string to a number representing the color channel
colorChannel = -1; // initialize to an invalid value
if (inputString == "R") {
colorChannel = 0;
} else if (inputString == "G") {
colorChannel = 1;
} else if (inputString == "B") {
colorChannel = 2;
} else {
// Show an error message if the input is invalid
showMessage("Invalid input: " + inputString);
}

// Loop through each TIFF file in the input folder
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        // Open the image using the Bio-Formats Importer plugin
        run("Bio-Formats Importer", "open=[" + inputFolder + list[i] + "] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
  
        // Split the channels of the image
        run("Split Channels");
  
        // Get the titles of the split channel windows
        titles = getList("image.titles");

        // Select the window corresponding to the specified color channel
        selectWindow(titles[colorChannel]);  

        // Construct the output filename by appending the bit depth to the input filename
        outputFilename = replace(list[i], ".tif", "_" + bit_depth + "bit"+inputString+".tif");
        outputPath = inputFolder + outputFilename;

        // Save the selected color channel with the new filename
        saveAs("Tiff", outputPath);

        // Close all images to free up memory, even in batch mode
        close("*");
    }
}

setBatchMode(false);
