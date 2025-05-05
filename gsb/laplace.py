import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

# Load a grayscale image using PIL
# Replace 'image.jpg' with your own image file
image = Image.open('figs/waterfall.jpg').convert('L')  # 'L' for grayscale
image_np = np.array(image)

# Define the 3x3 Laplacian kernel (4-connected)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

# Apply convolution using scipy.ndimage
laplacian_result = ndimage.convolve(image_np, laplacian_kernel, mode='reflect')

# Plot the original and the Laplacian-filtered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Grayscale Image")
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Laplacian Edge Detection")
plt.imshow(laplacian_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

