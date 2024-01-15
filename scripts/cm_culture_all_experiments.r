# Install required packages if not installed
# install.packages(c("ggplot2", "patchwork", "tidyr", "stringr", "magick", "purrr", "dplyr", "tidyverse", "httr", "httpgd", "BH"))

# Load required libraries
library(ggplot2)
library(patchwork)
library(tidyr)
library(stringr)
library(magick)
library(purrr)
library(dplyr)
library(tidyverse)
library(httpgd)
library(ggsignif)



# Function to read and process CSV files
read_and_process_csv <- function(csv_file) {
  tryCatch({
    path <- dirname(dirname(csv_file)) 
    subdir <- str_split(path, "/")[[1]][length(str_split(path, "/")[[1]])]
    experiment <- as.integer(str_extract(subdir, "\\d{2}"))
    print(basename(dirname(csv_file)))
    subdata_frame <- readr::read_csv(csv_file) %>%
      mutate(
             experiment = experiment,
             well = as.integer(str_match(filename, ".*well_(\\d+).*")[, 2]),
             tile = as.integer(str_match(filename, ".*tile_(\\d+).*")[, 2]),

      )
    if (nrow(subdata_frame) > 1) {
      return(subdata_frame)
    } else {
      cat("Excluding file due to only one row:", csv_file, "\n")
      return(NULL)
    }
  }, error = function(e) {
    cat("Error in file:", csv_file, "\n")
    print(e)
    return(NULL)
  })
}
#read_and_process_csv(csv_files[1])

# Function to add group column
add_group_col <- function(data) {
  updated_data <- data %>%
    mutate(group = case_when(
      experiment == 1 & well %in% c(1, 7) ~ "ctrl (PBS)",
      experiment == 1 & well %in% c(2, 8) ~ "ctrl (DMSO)",
      experiment == 1 & well %in% c(3, 9) ~ "4 uM CHIR99021",
      experiment == 1 & well %in% c(4, 10) ~ "MFAP5 50 ng/mL",
      experiment == 1 & well %in% c(5, 11) ~ "MFAP5 100 ng/mL",
      experiment == 1 & well %in% c(6, 12) ~ "MFAP5 200 ng/mL",
      experiment == 2 & well %in% c(1, 7) ~ "ctrl (PBS)",
      experiment == 4 & well %in% c(1, 7) ~ "untreated",
      experiment %in% c(2,4) & well %in% c(2, 8) ~ "ctrl (DMSO)",
      experiment %in% c(2,4) & well %in% c(3,9) ~ "4 uM CHIR99021",
      experiment == 2 & well %in% c(4,10) ~ "MFAP5 50 ng/mL",
      experiment == 4 & well %in% c(4,10) ~ "ctrl (PBS)",
      experiment == 2 & well %in% c(5,11) ~ "MFAP5 100 ng/mL",
      experiment == 2 & well %in% c(6,12) ~ "MFAP5 200 ng/mL",
      experiment == 4 & well %in% c(5,11) ~ "MFAP5 200 ng/mL",
      experiment == 4 & well %in% c(6,12) ~ "MFAP5 400 ng/mL",
      experiment == 5 & well == 1 ~ "untreated",
      experiment == 5 & well == 2 ~ "ctrl (DMSO)",
      experiment == 5 & well == 3 ~ "4 uM CHIR99021",
      experiment == 5 & well == 4 ~ "BSA (PBS)",
      experiment == 5 & well == 5 ~ "MFAP5 200 ng/mL",
      (experiment %in% c(6, 73, 76)) & well %in% c(1) ~ "untreated",
      (experiment %in% c(6, 73, 76)) & well %in% c(2) ~ "BSA (PBS)",
      (experiment %in% c(6, 73, 76)) & well %in% c(3) ~ "ctrl (DMSO)",
      (experiment %in% c(6, 73, 76)) & well %in% c(4) ~ "4 uM CHIR99021",
      (experiment %in% c(6, 73, 76)) & well %in% c(5) ~ "MFAP5 50 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(6) ~ "MFAP5 400 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(7) ~ "SPON2 200 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(8) ~ "SPON2 400 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(9) ~ "NTN1 250 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(10) ~ "NTN1 500 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(11) ~ "COMP 200 ng/mL",
      (experiment %in% c(5, 6, 73, 76)) & well %in% c(12) ~ "COMP 400 ng/mL",
      experiment == 8 & well %in% c(1,3,5,7,9,11) ~ "ctrl (DMSO)",
      experiment == 8 & well %in% c(2,4,6,8,10,12) ~ "4 uM CHIR99021",
      experiment == 10 & well %in% c(1) ~ "BSA (PBS)",
      experiment == 10 & well %in% c(2) ~ "ctrl (DMSO)",
      experiment == 10 & well %in% c(3) ~ "4 uM CHIR99021",
      experiment == 10 & well %in% c(4) ~ "MFAP5 50 ng/mL",
      experiment == 10 & well %in% c(5) ~ "MFAP5 400 ng/mL",
      experiment == 10 & well %in% c(6) ~ "SPON2 200 ng/mL",
      experiment == 10 & well %in% c(7) ~ "SPON2 400 ng/mL",
      experiment == 10 & well %in% c(8) ~ "NTN1 250 ng/mL",
      experiment == 10 & well %in% c(9) ~ "NTN1 500 ng/mL",
      experiment == 10 & well %in% c(10) ~ "COMP 200 ng/mL",
      experiment == 10 & well %in% c(11) ~ "COMP 400 ng/mL",
      experiment == 10 & well %in% c(12) ~ "untreated",
      experiment == 11 & well == 2 ~ "BSA (PBS)",
      experiment == 11 & well == 3 ~ "ctrl (DMSO)",
      experiment == 11 & well == 4 ~ "4 uM CHIR99021",
      experiment == 11 & well %in% c(5,6) ~ "SPON2 200 ng/mL",
      experiment == 11 & well %in% c(7,8) ~ "NTN1 500 ng/mL",
      experiment == 12 & well == 1 ~ "untreated",
      experiment == 12 & well == 2 ~ "BSA (PBS)",
      experiment == 12 & well == 3 ~ "ctrl (DMSO)",
      experiment == 12 & well == 4 ~ "4 uM CHIR99021",
      experiment == 12 & well == 5 ~ "MFAP5 50 ng/mL",
      experiment == 12 & well == 6 ~ "COMP 200 ng/mL",
      experiment == 13 & well == 1 ~ "untreated",
      experiment == 13 & well == 2 ~ "BSA (PBS)",
      experiment == 13 & well == 3 ~ "ctrl (DMSO)",
      experiment == 13 & well == 4 ~ "4 uM CHIR99021",
      experiment == 13 & well == 5 ~ "MFAP5 50 ng/mL",
      experiment == 13 & well == 6 ~ "COMP 200 ng/mL",
      TRUE ~ "NA"
    ))
  
  return(updated_data)
}

# Function to add density column
add_density_col <- function(data) {
  updated_data <- data %>%
    mutate(density = case_when(
      (experiment %in% c(1,2)) & well %in% c(1:6) ~ "10K",
      (experiment %in% c(1,2)) & well %in% c(7:12) ~ "20K",
      experiment %in% c(4,5) & well %in% c(1:12) ~ "20K",
      (experiment %in% c(6, 73, 76)) & well %in% c(1:12) ~ "30K",
      experiment == 8 & well %in% c(1, 2) ~ "30K",
      experiment == 8 & well %in% c(3, 4) ~ "40K",
      experiment == 8 & well %in% c(5, 6) ~ "50K",
      experiment == 8 & well %in% c(7, 8) ~ "60K",
      experiment == 8 & well %in% c(9, 10) ~ "70K",
      experiment == 8 & well %in% c(11, 12) ~ "80K",
      experiment %in% c(10,11,12,13) & well %in% c(1:12) ~ "30K",
      TRUE ~ "NA"
    ))
  
  return(updated_data)
}

# Function to save filtered data to CSV
save_filtered_data <- function(data, output_path) {
  write.csv(data, file = output_path, row.names = FALSE)
}

# Main script
parent_directory_path <- "/run/user/1000/gvfs/smb-share:server=anaentreg22.local,share=disk2/Asli/Experiments"
output_path <- "/run/user/1000/gvfs/smb-share:server=anaentreg22.local,share=disk2/Asli/all_experiments.csv"

csv_files <- list.files(parent_directory_path, pattern = ".csv", full.names = TRUE, recursive = TRUE)

# remove all_experiments.csv from csv_files if it exists
csv_files <- csv_files[!grepl("all_experiments.csv", csv_files)]


data_frames <- lapply(csv_files, read_and_process_csv) %>% purrr::compact()

data_frame <- bind_rows(data_frames)

colnames(data_frame) 


filtered_data <- data_frame %>%
  filter(DAPI_centroid_in_FITC_bbox == TRUE) %>%
  mutate(cell_id = paste0(label, experiment, well, tile)) %>%
  add_group_col() %>%
  add_density_col() %>%
  group_by(cell_id) %>%
  mutate(nuclei_count = sum(ifelse(DAPI_centroid_in_FITC_bbox, TRUE, FALSE))) %>%
  mutate(tritc_pos_nuclei_count = sum(ifelse(TRITC_centroid_in_DAPI_bbox, TRUE, FALSE))) %>%
  ungroup() %>%
  select(filename,cell_id, tile, well, group, density, experiment, area_um2, eccentricity, mean_intensity, nuclei_count, tritc_pos_nuclei_count) %>%
  distinct()




save_filtered_data(filtered_data, output_path)
