import cv2
import numpy as np

input_img = cv2.imread("/home/hqcthunders/Downloads/bsx.jpg")

imgGray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(imgGray, 9, 75, 75) # if sigma values < 10 or > 150, we couldn't get good result
equal_histogram = cv2.equalizeHist(noise_removal)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations = 20)
sub_morph_image = cv2.subtract(equal_histogram, morph_image)

ret, thresh_image = cv2.threshold(sub_morph_image, 0, 255, cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image, 250, 255)

kernel = np.ones((3, 3), np.uint8)

dilated_image = cv2.dilate(canny_image, kernel, iterations = 1)

new, contours, hierachy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

img_contour = cv2.drawContours(input_img, [screenCnt], -1, (0, 255, 0), 3)

(x, y, w, h) = cv2.boundingRect(screenCnt)
roi = input_img[y:y+h, x:x+w]
roi1 = roi.copy()

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 1)
ret, thre = cv2.threshold(roi_blur, 120, 255, cv2.THRESH_BINARY_INV)
kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(thre, cv2.MORPH_DILATE, kerel3)
_, cont, hier = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

areas_ind = {}
areas = []
for ind, cnt in enumerate(cont):
    area = cv2.contourArea(cnt)
    areas_ind[area] = ind
    areas.append(area)
areas = sorted(areas, reverse=True)[2:10]
for i in areas:
    (x, y, w, h) = cv2.boundingRect(cont[areas_ind[i]])
    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

character = []
for i in areas:
    (x, y, w, h)= cv2.boundingRect(cont[areas_ind[i]])
    image = roi[y:y+h, x:x+w]
    character.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print(character[0])
cv2.imshow("Image Gray", cv2.resize(imgGray, (400, 400)))
cv2.imshow("Noise Removal", cv2.resize(noise_removal, (400, 400)))
cv2.imshow("Equal histogram", cv2.resize(equal_histogram, (400, 400)))
cv2.imshow("Morph Image", cv2.resize(morph_image, (400, 400)))
cv2.imshow("Sub morph", cv2.resize(sub_morph_image, (400, 400)))
cv2.imshow("Canny Image", cv2.resize(canny_image, (400, 400)))
cv2.imshow("Dilated Image", cv2.resize(dilated_image, (400, 400)))
cv2.imshow("Contours", cv2.resize(img_contour, (400, 400)))
cv2.imshow("ROI", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
cv2.imshow("Thre Mor", thre_mor)
cv2.imshow("ROI 1", roi1)
cv2.imwrite("bsx.jpg", roi)
# cv2.imshow("IM", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
