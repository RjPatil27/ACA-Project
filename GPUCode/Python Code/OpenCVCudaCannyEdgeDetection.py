import numpy as np
import cv2 as cv
from pathlib import Path
import time

# taking path of image files from the folder
path = Path(".")
path = path.glob("Dataset/*.jpg")

count = 0
totalTime = 0

# declare memory on GPU
src = cv.cuda_GpuMat()

for imagepath in path:

    # imread() reads single image from folder
    src_host = cv.imread(str(imagepath), cv.IMREAD_GRAYSCALE)
    # transferring data to GPU memory
    src.upload(src_host)

    start_t = time.time()
    
    # performing guassian blur to reduce noise
    gaussian_filter = cv.cuda.createGaussianFilter(src.type(), src.type(), (3, 3), 0, cv.BORDER_DEFAULT)
    gpu_blur = gaussian_filter.apply(src)

    # creating canny edge detector in GPU
    canny_egde_detector = cv.cuda.createCannyEdgeDetector(50, 100)
    # detecting edges in GPU
    gpu_canny = canny_egde_detector.detect(gpu_blur)
    
    totalTime = totalTime + (time.time() - start_t)

    # transferring data back to CPU memory
    result_host = gpu_canny.download()
    
    count += 1;
    
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)
    
print("ImageCount =", count, "\nTimeRequired =", totalTime)
