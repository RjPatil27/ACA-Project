import cv2
from pathlib2 import Path
import time

#taking path of image files from the folder
path = Path(".")
path = path.glob("*.jpg")
count = 0
totalTime=0

for imagepath in path:
    # imread() reads single image from folder
    start = time.time()
    img = cv2.imread(str(imagepath))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # GaussianBlur() function convert image into Blur image which reduce noise from the image.
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # Laplacian() function applies Laplacian edge algorithm on Gaussian blurred image.
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    # plt.imshow(laplacian,cmap='gray')
    # cv2.imwrite("Laplacian_" + str(count) + ".jpg", laplacian)
    count+=1
    totalTime = totalTime + time.time() - start
    cv2.waitKey(0)

end = time.time()

print("ImageCount = ",count,"\nTimeRequired =",totalTime)


