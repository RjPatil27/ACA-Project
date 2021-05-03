import cv2
import numpy as np
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")

images = []
count = 0

for imagepath in path:
    img = cv2.imread(str(imagepath))
    imS = cv2.resize(img, (940, 600))
    images.append(imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    # cv2.imshow("Prewitt", img_prewittx + img_prewitty)
    count+=1
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",end-start)
