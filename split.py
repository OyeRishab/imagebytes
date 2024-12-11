from PIL import Image
import os

def create_image_dataset(large_image_path, output_dir):
    # Open the large image
    large_image = Image.open(large_image_path)
    width, height = large_image.size
    
    # Define the size of the small images
    crop_size = 256
    
    # Calculate the number of crops in both dimensions
    num_crops_x = width // crop_size
    num_crops_y = height // crop_size
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through the large image and create crops
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            left = i * crop_size
            upper = j * crop_size
            right = left + crop_size
            lower = upper + crop_size
            
            # Crop the image
            crop = large_image.crop((left, upper, right, lower))
            
            # Save the cropped image
            crop.save(os.path.join(output_dir, f'SAR6_{i}_{j}.png'))

# Example usage
create_image_dataset('SAR6.jpg', 'images/sar_train/')