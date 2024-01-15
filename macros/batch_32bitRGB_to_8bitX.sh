#!/bin/bash

echo "Enter color channel to extract (R, G, or B):"
read channel 

for file in *.tif; do
    convert "$file" -channel $channel -depth 8 -separate -set filename:base "%[basename]" "${file%.*}_8bit.tif"
done

