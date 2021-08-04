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

    # creating laplacian edge detector in GPU
    laplace_filter = cv.cuda.createLaplacianFilter(gpu_blur.type(), gpu_blur.depth(), 3)
    # applying laplacian edge detection in GPU
    gpu_laplace = laplace_filter.apply(gpu_blur)
    
    totalTime = totalTime + (time.time() - start_t)

#     abs_laplace = cv.cuda.abs(gpu_laplace)

    # transferring data back to CPU memory
    src_host = gpu_laplace.download()
    
    count += 1;
    
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)

print("ImageCount =", count, "\nTimeRequired =", totalTime)
