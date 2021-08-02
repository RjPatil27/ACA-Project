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
    src_host = cv.imread(str(imagepath))
    # transferring data to GPU memory
    src.upload(src_host)
    
    start_t = time.time()

    # performing bilateral filtering in GPU
    gpu_bilateral = cv.cuda.bilateralFilter(src, 7, 75, 75)
    
    totalTime = totalTime + (time.time() - start_t)

    # transferring data back to CPU memory
    result_host = gpu_bilateral.download()

    count += 1;
    
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)

print("ImageCount =", count, "\nTimeRequired =", totalTime)
