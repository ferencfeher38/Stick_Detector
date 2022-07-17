import math
import random
import cv2
import numpy as np

# Image scan:
img = cv2.imread('palcika4.jpg')
img3 = img.copy()
# cv2.imshow('img', img)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# Contrast and brightness:
contrast = 90
brightness = 50
x = np.arange(0, 256, 1)

factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
lut = np.uint8(np.clip(brightness + factor * (np.float32(x) - 128.0) + 128, 0, 255))

brightness_contrast = cv2.LUT(img, lut)
# cv2.imshow('brightness_contrast', brightness_contrast)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# Convert img to gray:
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', img2)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# Blur:
blur = cv2.GaussianBlur(img2,  # image
                        (11, 11),  # tuple, mask size
                        0)  # sigmaX
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# Threshold:
threshold = np.ndarray(blur.shape, blur.dtype)

threshold[blur >= 75] = 255
threshold[blur < 75] = 0
# cv2.imshow("threshold", threshold)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# TEST1: Wrong threshold:
# threshold2 = np.ndarray(blur.shape, blur.dtype)

# threshold2[blur >= 66] = 255
# threshold2[blur < 66] = 0
# cv2.imshow("threshold2", threshold2)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# TEST2: Only thinning:
# thn2 = cv2.ximgproc.thinning(img2, None, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
# cv2.imshow('thinning2', thn2)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# TEST3: Noise:
def add_noise(img):
    row, col = img.shape

    number_of_pixels = random.randint(100, 1000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    return img


#add_noise(threshold)
# ----------------------------------------------------------------------------------------

# Thinning:
thn = cv2.ximgproc.thinning(threshold, None, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
# cv2.imshow('thinning', thn)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilate = cv2.dilate(thn, kernel, iterations=1)
# cv2.imshow("dilate", dilate)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------

# 1. Line detection and draw:
lines = cv2.HoughLines(thn,  # image
                       1,  # pixel
                       np.pi / 180,  # angel
                       140)  # threshold

sizemax = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
        pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
        cv2.line(img, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)
# ----------------------------------------------------------------------------------------

# 2. Line detection and draw:
linesP = cv2.HoughLinesP(thn,  # image
                         1,  # pixel
                         np.pi / 180,  # angle
                         140,  # threshold
                         None,
                         50,  # minLineLength
                         50)  # maxLineGap

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img3, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
# ----------------------------------------------------------------------------------------

# Display the results:
print('1. Sticks on the image:', len(lines))
cv2.imshow("1._detected_lines", img)
cv2.waitKey(0)

print('2. Sticks on the image:', len(linesP))
cv2.imshow("2._detected_lines", img3)
cv2.waitKey(0)

cv2.destroyAllWindows()
