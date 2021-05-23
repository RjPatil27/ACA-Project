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
    # GaussianBlur() function convert image into Blur image which reduce noise from the image.
    Gaussian = cv2.GaussianBlur(img,(3,3),0)
    # cv2.imshow('Gaussian Blurring', Gaussian)
    # cv2.imwrite("GaussianOP_" + str(count) + ".jpg", Gaussian)
    count+=1
    totalTime = totalTime + time.time() - start
    cv2.waitKey(0)

end = time.time()
print("ImageCount = ",count,"\nTimeRequired =",totalTime)
