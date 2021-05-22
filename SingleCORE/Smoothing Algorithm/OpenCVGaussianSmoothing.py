import cv2
from pathlib2 import Path
import time

start= time.time()
path = Path(".")

path = path.glob("*.jpg")

images = []
count=0
start = time.time()

for imagepath in path:
    img = cv2.imread(str(imagepath))
    # imS = cv2.resize(img, (940, 600))
    # images.append(imS)
    Gaussian = cv2.GaussianBlur(img,(3,3),0)
    # cv2.imshow('Gaussian Blurring', Gaussian)
    # cv2.imwrite("GaussianOP_" + str(count) + ".jpg", Gaussian)
    count+=1
    cv2.waitKey(0)

end = time.time()
print("ImageCount = ",count,"\nTimeRequired =",end-start)
