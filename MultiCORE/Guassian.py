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
    Gaussian = cv2.GaussianBlur(imS,(7,7),0)
    return Gaussian
    
def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]

def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    for imagePath in payload["input_paths"]:
        image = cv2.imread(imagePath)
        Gaussian = process(image)

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
    print("[INFO] Gaussian blur smoothing complete")

    end=time.time();
    print(f'Time to complete: {end-start}');

