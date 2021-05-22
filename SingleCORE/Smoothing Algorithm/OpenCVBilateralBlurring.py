import cv2
from pathlib2 import Path
import time

start = time.time()
path = Path(".")

path = path.glob("*.jpg")

# images = []
count=0
start = time.time()

for imagepath in path:
    img = cv2.imread(str(imagepath))
    # imS = cv2.resize(img, (940, 600))
    # images.append(imS)
    bilateral = cv2.bilateralFilter(img, 7, 75, 75)
    # cv2.imshow('Bilateral Blur', bilateral)
    # cv2.imwrite("BilateralOP_" + str(count) + ".jpg", bilateral)
    count+=1
    cv2.waitKey(0)
end = time.time()
print("ImageCount = ",count,"\nTimeRequired =",end-start)
