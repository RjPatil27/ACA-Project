import cv2
import matplotlib.pyplot as plt
from pathlib2 import Path

path = Path(".")

path = path.glob("*.jpg")

images = []

for imagepath in path:
    img = cv2.imread(str(imagepath),0)
    imS = cv2.resize(img, (940, 600))
    images.append(img)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(imS, (3, 3), 0)
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    plt.imshow(laplacian,cmap='gray')
    plt.show()


