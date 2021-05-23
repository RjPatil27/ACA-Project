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
    median = cv2.medianBlur(img, 3)
    # cv2.imshow('Median Blurring', median)
    # cv2.imwrite("MedianOP_" + str(count) + ".jpg", median)
    totalTime = totalTime + time.time() - start
    count+=1
    cv2.waitKey(0)

print("ImageCount = ",count,"\nTimeRequired =",totalTime)
