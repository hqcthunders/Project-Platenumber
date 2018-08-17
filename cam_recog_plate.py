import cv2
import numpy as np

from video import create_capture
from common import draw_str

def draw_contours(contour):
    contours = sorted(contour, key=cv2.contourArea, reverse=True)[:10]
    # screenCnt = none
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06*peri, True)
        if len(approx) == 4:
            return approx


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while(True):
        ret, image = cap.read()

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
        equal_histogram = cv2.equalizeHist(noise_removal)
        kerel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kerel, iterations=20)
        sub_morph_image = cv2.subtract(equal_histogram, morph_image)

        ret, threshold_img = cv2.threshold(sub_morph_image, 0, 255, cv2.THRESH_OTSU)
        canny_image = cv2.Canny(threshold_img, 250, 255)

        kerel = np.ones((3, 3), np.uint8)

        dilated_img = cv2.dilate(canny_image, kerel, iterations=1)
        new, contours, hierachy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        screenCnt = draw_contours(contours)
        img_contours = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        print(img_contours)
        cv2.imshow("Gray", img_contours)
        if cv2.waitKey(5) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
