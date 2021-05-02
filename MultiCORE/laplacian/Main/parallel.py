#import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def process(image):
   

    # Apply gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Positive Laplacian Operator
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    return laplacian
    
def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]




def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    for imagePath in payload["input_paths"]:
         # Open the image
        image = cv2.imread(imagePath)
        laplacian = process(image)
        plt.figure()
        filename=os.path.basename(imagePath)
        plt.title(filename)
        plt.imsave(filename+'.jpg', laplacian, cmap='gray')
        plt.imshow(laplacian, cmap='gray')
        plt.show()
