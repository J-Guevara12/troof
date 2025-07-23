find "Cityscape Dataset/Residential" -type f | head -n 20 | while read image; do 
    python inference.py "$image" 
done
