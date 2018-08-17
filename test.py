from PIL import Image
import pytesseract
import cv2
import numpy as np
import os

image = cv2.imread('nam.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
image_dilated = cv2.dilate(image_gray, kernel, iterations=1)
image_erode = cv2.erode(image_dilated, kernel, iterations=1)

img = cv2.adaptiveThreshold(image_erode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

cv2.imwrite('test.png', img)
text = pytesseract.image_to_string(Image.open('test.png'))
print(text)
#cv2.imshow("ima", image)
#cv2.waitKey(0)
