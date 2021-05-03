import cv2
from pathlib2 import Path

path = Path(".")

path = path.glob("*.jpg")

images = []

for imagepath in path:
    img = cv2.imread(str(imagepath),0)
    imS = cv2.resize(img, (940, 600))
    images.append(imS)

    img_sobelx = cv2.Sobel(imS, cv2.CV_8U, 1, 0, ksize=5)
    img_sobely = cv2.Sobel(imS, cv2.CV_8U, 0, 1, ksize=5)
    img_sobel = img_sobelx + img_sobely

    # cv2.imshow("Sobel X", img_sobelx)
    # cv2.imshow("Sobel Y", img_sobely)
    cv2.imshow("Sobel", img_sobel)

    cv2.waitKey(0)
