import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('house.jpg')

# Apply Canny
edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

plt.figure()
plt.title('House')
plt.imsave('Canny_House.jpg', edges, cmap='gray')
plt.imshow(edges, cmap='gray')
plt.show()
