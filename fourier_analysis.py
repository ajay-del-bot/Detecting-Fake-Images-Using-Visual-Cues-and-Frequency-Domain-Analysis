import cv2
import numpy as np
import matplotlib.pyplot as plt


image_paths = ['/content/drive/MyDrive/image (1).jpg', '/content/drive/MyDrive/image.jpg']  # Replace with your image paths

for idx, image_path in enumerate(image_paths):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} not found or path is incorrect")

   
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image {idx+1}')
    plt.axis('off')
    plt.show()

    
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    
    plt.figure(figsize=(6,6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'Magnitude Spectrum {idx+1}')
    plt.axis('off')
    plt.show()

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    r = 30  
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

   
    f_shift_masked = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_masked)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    
    plt.figure(figsize=(6,6))
    plt.imshow(img_filtered, cmap='gray')
    plt.title(f'Low-Pass Filtered Image {idx+1}')
    plt.axis('off')
    plt.show()

    
    f_ishift_original = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift_original)
    img_back = np.abs(img_back)

   
    plt.figure(figsize=(6,6))
    plt.imshow(img_back, cmap='gray')
    plt.title(f'Reconstructed Image {idx+1}')
    plt.axis('off')
    plt.show()
