import cv2 as cv
import numpy as np

window_name = 'Augmented Video Footage View'
file_name = 'D:/University/4th yr/DSC4013/Week 4/resources/video/COF.avi'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

source = cv.VideoCapture(file_name)   
success, image = source.read()
factor = 2
height, width, layers = image.shape

new_h = height // factor
new_w = width // factor

# Calculate gap size
gap = 20

while success and (cv.waitKey(1) & 0xFF != ord('q')):  # 27 for esc
    new_width = width * 2 + gap
    new_height = height * 2 + gap

    
    # edge detection
    edges = cv.Canny(image, 150, 150)
    edges_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # Gaussian blurring
    gauss_blur = cv.GaussianBlur(image, (5, 5), 0)

    # Median blurring
    median_blur = cv.medianBlur(image, 5)

    # Resize edges to match new dimensions
    resize = cv.resize(edges, (new_width - width, new_height - height), interpolation=cv.INTER_AREA)

    combined_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Place original image in top-left segment
    combined_img[:height, :width] = image.copy()

    # Place edges in top-right segment with gap
    combined_img[:height, width+gap:] = edges_color.copy()

    # Place Gaussian blurred image in bottom-left segment with gap
    combined_img[height+gap:, :width] = gauss_blur.copy()

    # Place median filtered image in bottom-right segment with gap
    combined_img[height+gap:, width+gap:] = median_blur.copy()

    cv.imshow(window_name, combined_img)
    success, image = source.read()

source.release()
cv.destroyAllWindows()

