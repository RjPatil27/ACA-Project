import cv2
from pathlib2 import Path
import time

#taking path of image files from the folder
path = Path(".")
path = path.glob("*.jpg")
count=0
totalTime=0

for imagepath in path:
    # imread() reads single image from folder
    start = time.time()
    img = cv2.imread(str(imagepath))
    # bilateralFilter() function applies Bilateral Blur algorithm on image.
    bilateral = cv2.bilateralFilter(img, 7, 75, 75)
    # cv2.imshow('Bilateral Blur', bilateral)
    # cv2.imwrite("BilateralOP_" + str(count) + ".jpg", bilateral)
    totalTime = totalTime + time.time() - start
    count+=1
    cv2.waitKey(0)

print("ImageCount = ",count,"\nTimeRequired =", totalTime)
