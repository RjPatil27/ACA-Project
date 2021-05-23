import cv2
from pathlib2 import Path
import time

#taking path of image files from the folder
path = Path(".")
path = path.glob("*.jpg")
count = 0
totalTime = 0

#loop for getting one by one image file as input for CANNY algorithm
for imagepath in path:
    # imread() reads single image from folder
    start = time.time()
    img = cv2.imread(str(imagepath), cv2.IMREAD_GRAYSCALE)
    # GaussianBlur() function convert image into Blur image which reduce noise from the image.
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    # Canny() function from OpenCV applies Canny algorithm on Gaussian Image.
    img_canny = cv2.Canny(blur_img,50,100)
    totalTime = totalTime + time.time() - start
    # cv2.imshow("Canny", img_canny)
    # cv2.imwrite("OP"+str(count)+".jpg",img_canny)
    count+=1
    cv2.waitKey(0)

print("ImageCount = ",count,"\nTimeRequired =", totalTime)
