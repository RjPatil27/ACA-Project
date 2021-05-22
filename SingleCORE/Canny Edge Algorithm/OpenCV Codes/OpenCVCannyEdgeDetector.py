import cv2
import numpy as np
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")
count=0
# images = []

for imagepath in path:
    img = cv2.imread(str(imagepath))
    # imS = cv2.resize(img, (940, 600))
    # images.append(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_canny = cv2.Canny(blur_img,50,100)
    # cv2.imshow("Canny", img_canny)
    # cv2.imwrite("OP"+str(count)+".jpg",img_canny)
    count+=1
    cv2.waitKey(0)

end = time.time()
print("ImageCount = ",count,"\nTimeRequired =",end-start)
