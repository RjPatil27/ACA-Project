import cv2
import numpy as np

image = cv2.imread('apple.jpg')

cv2.imshow('Original Image', image)
cv2.waitKey(0)

#Guassian Blurr
Gaussian = cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.imwrite('GaussianResult.jpg',Gaussian)
cv2.waitKey(0)

#Median Blur
median = cv2.medianBlur(image,5)
cv2.imshow('Median Blurring', median)
cv2.imwrite('MedianResult.jpg',median)
cv2.waitKey(0)

#Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blur', bilateral)
cv2.imwrite('BilateralResult.jpg',bilateral)
cv2.waitKey(0)
