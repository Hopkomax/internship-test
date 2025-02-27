import os
import cv2
from PIL import Image

# Path to the image dataset
data_path = "F:/gitRepo/internship-test/task2_NER_imageClassification/data/images"

# Function to check and remove corrupted images
def check_and_clean_images(dataset_path):
    corrupted_files = []
    
    # Loop through each animal folder
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):  # Ensure it's a folder
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Check if file is a valid image
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify image integrity
                        img.close()
                    
                    # Open image with OpenCV to check further
                    img_cv = cv2.imread(img_path)
                    if img_cv is None:
                        raise Exception("Corrupted Image")

                except Exception as e:
                    print(f"❌ Corrupted file: {img_path} - {e}")
                    corrupted_files.append(img_path)
    
    # Remove corrupted images
    for file in corrupted_files:
        os.remove(file)
        print(f"🗑️ Deleted: {file}")

    print("\n✅ Dataset cleaning complete!")

# Run cleaning function
check_and_clean_images(data_path)
