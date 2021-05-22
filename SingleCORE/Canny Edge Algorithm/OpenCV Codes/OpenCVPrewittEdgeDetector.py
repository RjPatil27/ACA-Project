import cv2
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")
count=0
images = []

for imagepath in path:
    img = cv2.imread(str(imagepath))
    # imS = cv2.resize(img, (940, 600))
    # images.append(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_sobelx = cv2.Sobel(blur_img, cv2.CV_8U, 0, 1, ksize=3)
    img_sobely = cv2.Sobel(blur_img, cv2.CV_8U, 1, 0, ksize=3)
    img_sobel = img_sobelx + img_sobely

    # cv2.imshow("Sobel X", img_sobelx)
    # cv2.imshow("Sobel Y", img_sobely)
    # cv2.imshow("Sobel", img_sobel)
    # cv2.imwrite("SobelOP_" + str(count) + ".jpg", img_sobel)
    count+=1
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",end-start)
