#import packages
from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import time
def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# GaussianBlur() function convert image into Blur image which reduce noise from the image.
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
# Horizontal direction mask and Vertical direction mask which goes as input to filter2D() function
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# filter2D() function applies Prewitt edge algorithm on Gaussian blur image.
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    return  img_prewittx + img_prewitty
    
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
        Prewitt = process(image)
        # cv2.imshow("Prewitt", Prewitt)
        # cv2.imwrite("imagePath" + ".jpg", Prewitt)

# Start time calculation
start=time.time();
# check to see if this is the main thread of execution
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # input image path given here in default
    ap.add_argument("-i", "--images", type=str, default=r'C:\Users\Admin\Desktop\Project\Programs\Dataset',
        help="path to input directory of images")
    ap.add_argument("-p", "--procs", type=int, default=0,
        help="# of processes to spin up")
    args = vars(ap.parse_args())
# determine the number of concurrent processes to launch and distributing load accross the system then create the list of process ID's
    procs = args["procs"] if args["procs"] > 0 else cpu_count()
    procIDs = list(range(0, procs))
    print("[INFO] grabbing image paths...")
# grab the paths to the input images, then determine the number of images each process will handle
    allImagePaths = sorted(list(paths.list_images(args["images"])))
    numImagesPerProc = len(allImagePaths) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
# chunk the image paths into N (approximately) equal sets, one set of image paths for each individual process
    chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))
# initialize the list of payloads
    payloads = []
# loop over the set chunked image paths
    for (i, imagePaths) in enumerate(chunkedPaths): 
# construct a dictionary of data for the payload, then add it to the payloads list
        data = {
            "id": i,
            "input_paths": imagePaths,
        }
        payloads.append(data)

# construct and launch the processing pool
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
# close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] Prewitt edge detection complete")

    end=time.time();
# Print time taken for processing
    print(f'Time to complete: {end-start}');

