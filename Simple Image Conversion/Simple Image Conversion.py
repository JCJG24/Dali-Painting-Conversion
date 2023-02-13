import cv2
import numpy as np

def dali_style(image):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # apply dilation to the edges to make them thicker
    dilation_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, dilation_kernel, iterations=1)

    # invert the colors of the edges to get white edges on a black background
    edges = cv2.bitwise_not(edges)

    # color the edges with a random pastel color to give a painted effect
    color = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)
    color = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * color
    dali_style = cv2.bitwise_or(color, color, mask=edges[..., None])

    return dali_style



# Load the input image
image = cv2.imread(r"C:\Users\JoJo\Pictures\Cyberpunk 2077\partygirls.png")

# Convert the image to a Salvador Dali-style painting
dali_style = dali_style(image)

# Save the output image
cv2.imwrite(r"C:\Users\JoJo\Pictures\Cyberpunk 2077\output.jpg", dali_style)
