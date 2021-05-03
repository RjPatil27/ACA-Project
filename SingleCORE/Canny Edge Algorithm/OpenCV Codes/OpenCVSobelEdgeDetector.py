import cv2
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")
count=0
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
    # cv2.imshow("Sobel", img_sobel)
    count+=1
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",end-start)
