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

    # imread() reads single image in grayscale format from folder
    src_host = cv.imread(str(imagepath), cv.IMREAD_GRAYSCALE)
    # transferring data to GPU memory
    src.upload(src_host)

    start_t = time.time()
    
    # performing guassian blur to reduce noise
    gaussian_filter = cv.cuda.createGaussianFilter(src.type(), src.type(), (3, 3), 0, cv.BORDER_DEFAULT)
    gpu_blur = gaussian_filter.apply(src)

    # x-axis kernel (horizontal direction mask)
    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # y-axis kernel (vertical direction mask)
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    # create horizontal mask filter
    img_prewittX = cv.cuda.createLinearFilter(gpu_blur.type(), gpu_blur.type(), kernel_x, (-1, -1))
    # apply horizontal mask filter
    gpu_prewittX = img_prewittX.apply(gpu_blur)
    # create vertical mask filter
    img_prewittY = cv.cuda.createLinearFilter(gpu_blur.type(), gpu_blur.type(), kernel_y, (-1, -1))
    # apply vertical mask filter
    gpu_prewittY = img_prewittY.apply(gpu_blur)

    # adding horiontal and vertical output
    gpu_prewitt = cv.cuda.add(gpu_prewittX, gpu_prewittY)

#     gpu_prewitt = cv.cuda.addWeighted(gpu_prewittX, 0.5, gpu_prewittY, 0.5, 0)
    
    totalTime = totalTime + (time.time() - start_t)
    
    # transferring data back to CPU memory
    result_host = gpu_prewitt.download()
    
    count += 1;
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)

print("ImageCount =", count, "\nTimeRequired =", totalTime)
