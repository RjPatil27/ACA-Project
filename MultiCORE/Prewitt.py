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
    imS = cv2.resize(image, (940, 600))
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    return  img_prewittx + img_prewitty
    
def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]

def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    for imagePath in payload["input_paths"]:
        image = cv2.imread(imagePath)
        Prewitt = process(image)

start=time.time();
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # input image path given here in default
    ap.add_argument("-i", "--images", type=str, default=r'C:\Users\Admin\Desktop\Project\my\edit\code\sample',
        help="path to input directory of images")
    ap.add_argument("-p", "--procs", type=int, default=0,
        help="# of processes to spin up")
    args = vars(ap.parse_args())

    procs = args["procs"] if args["procs"] > 0 else cpu_count()
    procIDs = list(range(0, procs))
    print("[INFO] grabbing image paths...")
    allImagePaths = sorted(list(paths.list_images(args["images"])))
    numImagesPerProc = len(allImagePaths) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

    payloads = []
    for (i, imagePaths) in enumerate(chunkedPaths): 
        data = {
            "id": i,
            "input_paths": imagePaths,
        }
        payloads.append(data)

    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] Prewitt edge detection complete")

    end=time.time();
    print(f'Time to complete: {end-start}');

