import numpy as np
import cv2
import os
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def calculate_mean_dct_spectrum(folder_path, image_size=(128, 128), max_images=10000):
    total_dct = None
    image_count = 0

    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # Read and resize the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            
            # Compute the DCT of the image
            img_dct = dct(dct(img.T, norm="ortho").T, norm="ortho")
            img_dct = np.abs(img_dct)  # Take the magnitude of the DCT

            # Initialize the total DCT accumulator on the first image
            if total_dct is None:
                total_dct = np.zeros_like(img_dct, dtype=np.float64)

            # Accumulate the DCT values
            total_dct += img_dct
            image_count += 1

        # Optional: Stop after max_images
        if image_count >= max_images:
            break

    print(image_count)
    # Compute the mean DCT spectrum
    mean_dct = total_dct / image_count if image_count > 0 else None
    return mean_dct

def plot_dct_spectrum(dct_spectrum):
    # Apply log scale for better visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log(dct_spectrum + 1e-8), cmap='viridis')
    plt.colorbar(label="Log Intensity")
    plt.title("Mean DCT Spectrum of Images")
    plt.xlabel("Frequency X-axis")
    plt.ylabel("Frequency Y-axis")
    plt.show()

# Example usage
#Deepfloyd_Outdoor
folder_path = 'dataset/validation_1'
mean_dct_spectrum = calculate_mean_dct_spectrum(folder_path)

if mean_dct_spectrum is not None:
    plot_dct_spectrum(mean_dct_spectrum)
else:
    print("No images were processed.")
