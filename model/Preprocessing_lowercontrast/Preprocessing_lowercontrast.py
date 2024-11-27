#This python program has been created to lower contrast the images in existing dataset, simulating foggy weather to train the model to increase the accuracy.
from PIL import Image, ImageEnhance
import os

def adjust_contrast(image_path, contrast_factor):
    with Image.open(image_path) as img:
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(contrast_factor)
        return img_enhanced

def process_images(input_folder, output_folder, contrast_factor=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = adjust_contrast(input_path, contrast_factor)
            img.save(output_path)
            print(f"Processed: {filename} -> saved to {output_path}")

if __name__ == "__main__":
    input_folder = ".\\input_images"
    output_folder = ".\\output_images"
    contrast_factor = 0.3
    process_images(input_folder, output_folder, contrast_factor)
