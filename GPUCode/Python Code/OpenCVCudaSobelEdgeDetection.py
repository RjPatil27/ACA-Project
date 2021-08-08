
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

    # derivative along the X axis
    gradX_filter = cv.cuda.createSobelFilter(gpu_blur.type(), gpu_blur.depth(), 0, 1, 3)
    grad_x = gradX_filter.apply(gpu_blur)
    # derivative along the X axis
    gradY_filter = cv.cuda.createSobelFilter(gpu_blur.type(), gpu_blur.depth(), 1, 0, 3)
    grad_y = gradY_filter.apply(gpu_blur)

    # adding x-axis and y-axis gradients 
    gpu_sobel = cv.cuda.add(grad_x, grad_y)
    
#     abs_grad_x = cv.cuda.abs(grad_x)
#     abs_grad_y = cv.cuda.abs(grad_y)

#     gpu_sobel = cv.cuda.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    totalTime = totalTime + (time.time() - start_t)

    # transferring data back to CPU memory
    result_host = gpu_sobel.download()
    
    count += 1;
#     cv.imwrite("output/OP" + str(count) + ".jpg", result_host)

print("ImageCount =", count, "\nTimeRequired =", totalTime)
