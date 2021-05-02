import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('house.jpg')

# Apply gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply gaussian blur
blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

# Positive Laplacian Operator
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)

plt.figure()
plt.title('LaplacianHouse')
plt.imsave('laplacian_house.jpg', laplacian, cmap='gray')
plt.imshow(laplacian, cmap='gray')
plt.show()