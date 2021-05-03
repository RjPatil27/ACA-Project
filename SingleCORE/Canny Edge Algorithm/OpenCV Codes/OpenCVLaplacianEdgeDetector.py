import cv2
import matplotlib.pyplot as plt
from pathlib2 import Path
import time

start = time.time()

path = Path(".")

path = path.glob("*.jpg")
count = 0
images = []

for imagepath in path:
    img = cv2.imread(str(imagepath),0)
    imS = cv2.resize(img, (940, 600))
    images.append(img)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(imS, (3, 3), 0)
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    plt.imshow(laplacian,cmap='gray')
    count+=1
    # plt.show()

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",end-start)


