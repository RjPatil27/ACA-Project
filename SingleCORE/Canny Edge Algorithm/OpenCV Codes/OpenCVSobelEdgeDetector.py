import cv2
from pathlib2 import Path
import time


#taking path of image files from the folder
path = Path(".")
path = path.glob("*.jpg")
count=0
totalTime = 0

for imagepath in path:
    # imread() reads single image from folder
    start = time.time()
    img = cv2.imread(str(imagepath))
    # cvtColor() function used to convert color image into Gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # GaussianBlur() function convert image into Blur image which reduce noise from the image.
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # Sobel() function from OpenCV applies Sobel Edge algorithm on blurred image
    img_sobelx = cv2.Sobel(blur_img, cv2.CV_8U, 0, 1, ksize=3)
    img_sobely = cv2.Sobel(blur_img, cv2.CV_8U, 1, 0, ksize=3)
    img_sobel = img_sobelx + img_sobely

    # cv2.imshow("Sobel X", img_sobelx)
    # cv2.imshow("Sobel Y", img_sobely)
    # cv2.imshow("Sobel", img_sobel)
    # cv2.imwrite("SobelOP_" + str(count) + ".jpg", img_sobel)
    totalTime = totalTime + time.time() - start
    count+=1
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",totalTime)
