# import packages
from Main.parallel import process_images
from Main.parallel import chunk
from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths
import numpy as np
import argparse
import os
import time

start=time.time();
# check to see if this is the main thread of execution
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=True, type=str,
		help="path to input directory of images")
	#ap.add_argument("-o", "--output", required=True, type=str,
	#	help="path to output directory to store intermediate files")
	ap.add_argument("-p", "--procs", type=int, default=-1,
		help="# of processes to spin up")
	args = vars(ap.parse_args())

	# determine the number of concurrent processes to launch when
	# distributing the load across the system, then create the list
	# of process IDs
	procs = args["procs"] if args["procs"] > 0 else cpu_count()
	procIDs = list(range(0, procs))
	# grab the paths to the input images, then determine the number
	# of images each process will handle
	print("[INFO] grabbing image paths...")
	allImagePaths = sorted(list(paths.list_images(args["images"])))
	numImagesPerProc = len(allImagePaths) / float(procs)
	numImagesPerProc = int(np.ceil(numImagesPerProc))
	# chunk the image paths into N (approximately) equal sets, one
	# set of image paths for each individual process
	chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

	# initialize the list of payloads
	payloads = []
	# loop over the set chunked image paths
	for (i, imagePaths) in enumerate(chunkedPaths):
		# construct the path to the output intermediary file for the
		# current process
		#outputPath = os.path.sep.join([args["output"],
		#	"proc_{}.pickle".format(i)])
		# construct a dictionary of data for the payload, then add it
		# to the payloads list
		data = {
			"id": i,
			"input_paths": imagePaths,
		#	"output_path": outputPath
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
	print("[INFO] multiprocessing complete")

	end=time.time();
	print(f'Time to complete: {end-start}');