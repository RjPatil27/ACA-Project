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
    
    # creating median blur filter in GPU
    median_filter = cv.cuda.createMedianFilter(src.type(), 7)
    # applying median blur in GPU
    gpu_median = median_filter.apply(src)
    
    totalTime = totalTime + (time.time() - start_t)

    # transferring data back to CPU memory
    result_host = gpu_median.download()
    
    count += 1;
    
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)

print("ImageCount =", count, "\nTimeRequired =", totalTime)
