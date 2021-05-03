import cv2
import numpy as np
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")
count=0
images = []

for imagepath in path:
    img = cv2.imread(str(imagepath))
    imS = cv2.resize(img, (940, 600))
    images.append(imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    # img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    img_canny = cv2.Canny(imS,100,200)
    # cv2.imshow("Canny", img_canny)
    count+=1
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",end-start)
