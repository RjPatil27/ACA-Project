import cv2
import numpy as np
from pathlib2 import Path
import time

#taking path of image files from the folder
path = Path(".")
path = path.glob("*.jpg")
count = 0
totalTime = 0

for imagepath in path:
    # imread() reads single image from folder
    start = time.time()
    img = cv2.imread(str(imagepath))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # GaussianBlur() function convert image into Blur image which reduce noise from the image.
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # Horizontal direction mask and Vertical direction mask which goes as input to filter2D() function
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # filter2D() function applies Prewitt edge algorithm on Gaussian blur image.
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    # cv2.imshow("Prewitt", img_prewittx + img_prewitty)
    # cv2.imwrite("SobelOP_" + str(count) + ".jpg", img_prewittx+img_prewitty)
    # cv2.imwrite("VerticalmaskOP_" + str(count) + ".jpg", img_prewittx)
    # cv2.imwrite("HorizontalmaskOP_" + str(count) + ".jpg",img_prewitty)
    totalTime = totalTime + time.time() - start
    count+=1
    cv2.waitKey(0)

print("ImageCount = ",count,"\nTimeRequired =", totalTime)
