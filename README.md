# ACA-Project

# Exploiting Parallelism in Image Processing Algorithms

## RAJAT PATIL
```
Department of CSEE
University of Maryland Baltimore Cty
Baltimore, Maryland
vk17971@umbc.edu
```
## ROHIT MOKASHI
```
Department of CSEE
University of Maryland Baltimore Cty
Baltimore, Maryland
rohitm3@umbc.edu
```
## NILAY FURSULE
```
Department of CSEE
University of Maryland Baltimore Cty
Baltimore, Maryland
nilayf1@umbc.edu
```
## Abstract
---
Various industries such as Computer Vision, Video Processing, Microscopic Imaging, Security Systems, \
Industrial Manufacturing uses Digital Image Processing. Many image processing applications require \
real-time image processing and analysis. It is very well known that image processing algorithms are \
resource-intensive and time-consuming.  Although CPUs nowadays are capable and powerful enough; \
performing image processing on a single-core CPU is uneconomical. For the last few years, \
there has been a new trend of  using multi-core architecture for performing image processing by \
exploiting parallelism in algorithms. The increasing computing power and programmability of \
multi-core architectures offer promising opportunities for parallelizing image processing algorithms. \
Significant performance gains over traditional CPU implementations can be achieved by taking advantage \
of the data, thread, and instruction parallelism offered by modern multi-core hardware. \
In this paper, we provide multi-core architecture-based methods to parallelize different image processing \
algorithms. First, this paper will briefly describe some image processing algorithms and details of the \
systems used for performing image processing. Later, it will elaborate on the single-core CPU, multi-core \
CPU, and the GPU-based approaches for image processing. Finally, this report will provide a detailed \
experimental analysis of the performance of different image processing algorithms by using \
the approaches mentioned above. Index Terms—Image Processing, Image Processing algorithm, Single-core, \
Multi-core, GPU, CUDA, NVIDIA, OpenCV, Multi-Processing, Computer architecture. 

---

## I. INTRODUCTION
---
Image processing algorithms are well-known for being computationally expensive 
and time-consuming. Although central processing units (CPUs) nowadays are capable
and powerful enough, performing image processing on a single-core CPU is 
uneconomical. As a result, over the last few decades, there has been a growing 
interest in developing and using multi-core architectures for image processing. 
Compared to central processing units (CPUs), we can get adequate acceleration
by using multi-core architecture to accelerate numerically intensive image processing 
algorithms. Image processing algorithms are categorized as low-level, intermediate-level, 
or high-level image processing based on the output generated [23]. 
In low-level image processing, the input provided to the algorithm is an image, and the output
generated is also an image. Smoothing, Sharpening, Filtering are examples of low-level 
image processing. In intermediate-level image processing, the input provided to the 
algorithm is an image, and the output is the attributes of an image. Motion Analysis, 
Object labeling are examples of intermediate-level image processing. In high-level 
image processing, the input provided to the algorithm is an image, and the output 
is the understanding of an image. Scene Understanding, Autonomous Navigation are 
examples of high-level image processing. The low-level image processing requires 
working on individual pixels, and the input image data is spatially localized.
Due to this, low-level image processing offers fine-grained parallelism. 
Hence, in this paper, our focus will be on low-level image processing algorithms. 
First, this paper will briefly describe low-level image processing algorithms and details of
the systems used for performing image processing. Further,this report will elaborate 
on the experimental analysis conducted by using three architectures: single-core CPU, multi-
core CPU, and GPU. Finally, this paper aims to provide the performance evaluation using 
a single-core approach and a multi-core approach.

---

## II. MOTIVATION
---

As mentioned earlier, CPUs nowadays are powerful and
capable enough to perform digital image processing. However,
using single-core CPUs to perform computationally expensive
image processing seems to be uneconomical. Multi-core archi-
tecture has emerged as a viable alternative to single-core CPUs
in recent years. Exploiting and accelerating parallel processing
has become an essential strategy for achieving better and more
competitive outcomes than single-core CPU implementations.
The use of multi-core architecture can obtain cost-effective
and energy-efficient implementations of image algorithms.

---

## III. DIGITAL IMAGE PROCESSING
---
Digital image processing is the method of manipulating
digital images using a computer. It is a signal and systems
sub-field that focuses primarily on images. DIP focuses on
the creation of a computer device that can process images.
A digital image is the system’s input, and it processes those
images using powerful algorithms to produce an image as an
output. Adobe Photo-shop is the most famous example. It is
one of the most commonly used applications for digital image
processing.

The following are the critical stages of image processing:
- Using image acquisition methods to import the image.
- Analyzing and editing the image.
- Output, which may be an altered image or a report based
    on the image’s analysis.
Enhancement of pictorial knowledge for human perception
is one of the goals of digital image processing. Furthermore,
image data processing for autonomous machine perception like
storage, transmission, and representation is another goal of
digital image processing.

---

### A. WHAT IS AN IMAGE?

---
A spatial representation of a two-dimensional or three-
dimensional scene is called an image. It’s a pixel array or ma-
trix that’s organized in columns and rows. A two-dimensional
array of rows and columns is also known as an image. Picture
components, image elements, and pixels make up a digital
image. A pixel is the most common unit of measurement for
the aspects of a digital image.

---

### B. IMAGE AS A MATRIX

---
In a screen, a digital gray-scale image is represented by
a pixel matrix. One matrix element—an integer from the
set—presents each pixel of such an image. Pixel presentation
numeric values are uniformly modified from zero (black pix-
els) to 255 (white pixels) (white pixels). The matrix represents
the color black as 0 and the color white as 1 in a binary image
of just two colors, i.e., black and white.
We have given a RGB matrix format in [fig 1],

---

![img1](https://github.com/RjPatil27/ACA-Project/blob/main/Images/Color%20Image%20formula.png)
```
Fig. 1. Matrix example of Image
```
### C. IMAGES TYPES

---
Images in various types,

- Binary Image - A binary image is made up of just two
    pixels: 0 and 1. The numbers 0 and 1 reflect black and
    white, respectively. Monochrome is another name for this.
- Black and White – The image are in black and white.
- 8 Bit Color Format – Also known as Gray-scale Picture,
    this format has 256 different shades of color. In this
    format, 0 represents black, 255 represents white, and 127
    represents Gray.
- 16 Bit Color Format – This format has a wide range of
    colors and is referred to as a High Color Format. Color
    distribution differs from that of a Gray-scale picture.

---

### D. PHASES OF IMAGE PROCESSING

---
The following are the basic steps in digital image process-
ing:
- Image Acquisition – Image acquisition entails pre-
    processing such as scaling and other adjustments. It may
    be as easy as receiving an image that has already been
    converted to digital format.
- Image Enhancement – Enhancement techniques, such as
    adjusting brightness and In contrast, bring out informa-
    tion that has been obscured and highlight those features
    of interest in an image.
- Image Restoration – Image restoration is the process
    of enhancing the quality of a picture. In the sense
    that restoration methods are usually based on statistical
    or probabilistic models of image deterioration, image
    restoration is objective.
- Color Image Processing – Due to the large increase in
    the use of digital images over the Internet, color im-
    age processing is becoming increasingly important. This
    could include color modeling and digital processing, for
    example. Enhancement, on the other hand, is a personal
    choice.
- Wavelets and Multi-resolution Processing (WAVELETS
    AND MULTI-RESOLUTION PROCESSING) Wavelets
    provide the basis for representing images in different
    degrees of resolution. For data compression, images are
    subdivided into smaller regions.
- Compression — Compression techniques minimize the
    amount of storage or bandwidth needed to store or
    transmit an image. Compressing data, particularly for use
    on the internet, is extremely important.
- Morphological Processing – Morphological processing
    extracts image components that are useful for describing
    and representing form.
- Segmentation — Segmentation is a technique for break-
    ing down an image into its component parts or artifacts.
    In general, one of the most challenging tasks in digital
    image processing is autonomous segmentation. A robust
    segmentation technique takes the process a long way
    toward solving imaging problems that involve individual
    object identification.
- Representation and Description — Representation and
    description almost always follow the performance of a
    segmentation level, which is typically raw pixel data that
    represents either a region’s boundary or all of its points.
    Extracting attributes that result in quantitative details of
    interest or are fundamental for distinguishing one class
    of objects from another is the subject of description.
- Object Recognition — Recognition is the method of
    assigning a label to an object based on its descriptors,
    such as ”apple.”

---
### E. DIP PROCESS FLOW
---
The simple diagram for understanding the digital image
processing working flow is given in [fig 2],

---
![img2](https://github.com/RjPatil27/ACA-Project/blob/main/Images/DIP%20Process%20Flow.png)
```
Fig. 2. Digital Image Processing process flow
```

### F. NEED OF AN DIGITAL IMAGE PROCESSING?

---
Image processing is often regarded as arbitrary image ma-
nipulation to meet an aesthetic ideal or support a preferred
truth. On the other hand, image processing is more precisely
characterized as a method of communication between the hu-
man visual system and digital imaging devices. Human vision
does not view the environment similarly as optical detectors
do, and display systems add to the noise and bandwidth
constraints.

---

## IV. IMAGE PROCESSING ALGORITHM
---
As we understand the Digital Image Processing concept and
why it is necessary for the market. In this section, we will get
to know about few image processing algorithms which we
used for analysis purpose in our project.

---

### A. EDGE DETECTION ALGORITHM

---

One of the most basic operations in image processing is
edge detection. It assists you in reducing the amount of data
(pixels) to process while maintaining the image’s ”structural”
aspect.
For our research, we used four different edge detection
algorithms. Below, we’ve covered a fundamental feature of
these algorithms.

---

#### 1) CANNY EDGE DETECTION ALGORITHM:
The Canny Edge detector is an edge detection algorithm that detects a
large range of edges in images using a multi-stage algorithm.
It was founded in 1986 by John F. Canny.
Canny edge detection is a technique for extracting useful
structural information from various vision artifacts while re-
ducing the amount of data to be processed drastically. It’s
been used in a variety of computer vision systems. Canny
discovered that the criteria for using edge detection on a
variety of vision systems are remarkably similar. As a result,
an edge detection solution that meets these criteria can be used
in a variety of scenarios.

The Canny edge detection algorithm is composed of 5 steps:
1) Noise reduction
2) Gradient calculation
3) Non-maximum suppression
4) Double threshold
5) Edge Tracking by Hysteresis

After applying these steps, [fig 3] shows original image and
[fig 4] shows canny image output.

![img3](https://github.com/RjPatil27/ACA-Project/blob/main/Images/bike1.jpg)
```
Fig. 3. ORIGINAL IMAGE
```
![img4](https://github.com/RjPatil27/ACA-Project/blob/main/Images/OP1.jpg)
```
Fig. 4. CANNY IMAGE OUTPUT
```

The algorithm is based on gray-scale images, which is a
significant point to note. As a result, the image must first be
converted to gray-scale images. As a result, before proceeding
with the above steps, you must first convert the picture to
gray-scale.

---

2) SOBEL EDGE DETECTION ALGORITHM:The Sobel
```
Edge Detection algorithm is an image edge detection algo-
rithm that employs the Sobel operator. Irwin Sobel and Gary
Feldman are the creators. The Sobel operator is based on
convolving the image in the horizontal and vertical directions
with a thin, separable, integer-valued filter, and thus is com-
putationally inexpensive.
The Sobel edge detector is dependent on gradients. It
operates with derivatives of the first order. The first derivatives
of the images are calculated separately for the X and Y axes.
The derivatives are all estimates.
The [fig 5] contains kernels used for convolution to approxi-
mate them:
The derivative along the X axis is approximated by the
kernel on the left. The Y axis is represented by the one on
the right.
```

```
Fig. 5. Sobel operator horizontal and vertical kernel example
```
We can calculate the following using this data:
1) Magnitude or “strength” of the edges
(G^2 x+G^2 y) (1)
2) Approximate strength:

|Gx|+|Gy| (2)
3) The orientation of the edge:
arctan(Gy/Gx) (3)
After applying these steps, we will get following result as
output [fig 6]:

```
Fig. 6. SOBEL IMAGE OUTPUT
```
Gray images are also compatible with the Sobel Edge
detection algorithm. As a result, before proceeding with the
above steps, you must first convert the picture to gray-scale.

3) LAPLACIAN EDGE DETECTION ALGORITHM: The
```
Laplacian Edge Detection algorithm uses the Laplacian Oper-
ator to detect edges. Second derivative operators are used to
detect Laplacian edges. It only has one kernel. The Laplacian
operator calculates second order derivatives in a single pass.
The kernel used in Laplacian edge detection is shown below
in [fig 7],
We can use one of these, or we can construct a 5*5 kernel
for a better approximation. Due to the fact that we are dealing
```
```
Fig. 7. Laplacian kernels example
```
```
with second order derivatives, the Laplacian edge is highly
sensitive to noise. Normally, we’d like to minimize noise;
maybe we can use Gaussian blur in this algorithm to overcome
this limitation.
After applying all required steps, we can see the output of
original image below in [fig 8],
```
```
Fig. 8. LAPLACIAN IMAGE OUTPUT
```
```
Laplacians are computationally faster and can achieve
excellent results in certain cases.
```

4) PREWITT EDGE DETECTION ALGORITHM:
```
The Prewitt Edge Detection Algorithm is an image edge detection
algorithm that employs the Prewitt operator. The Prewitt
operator distinguishes between two types of edges: horizontal
and vertical.
The difference between corresponding pixel intensities of
an image is used to measure edges. Derivative masks refer to
all of the masks that are used for edge detection. Since an
image is a signal, only differentiation can be used to quantify
signal changes. As a result, these operators are also known as
derivative masks or derivative operators.
The following properties should be present in all derivative
marks:
- The mask should have the opposite sign.
- The sum of the masks should be zero.
- More weight translates to better edge detection.
The Prewitt operator gives us two masks: one for detecting
horizontal edges and the other for detecting vertical edges.
```
VERTICAL DIRECTION MASK:

Example of vertical direction mask is given in [fig 9].
```
```
Fig. 9. Vertical direction mask example
```
Since the zero’s column is in the vertical direction, the above
mask would find the edges in the vertical direction. When you
convolve this mask on an image, it will reveal the image’s
vertical edges.
WORKING OF VERTICAL DIRECTION MASK: When
we apply this mask to an image, the vertical edges become
more prominent. It simply calculates the difference of pixel
intensities in an edge field, similar to a first order derivative.
Since the center column is zero, it does not have the image’s
original values, instead calculating the difference between the
right and left pixel values along that edge. The edge strength
is increased, and the picture is enhanced in comparison to the
original.

HORIZONTAL DIRECTION MASK:

Example of horizontal direction mask is given in [fig
10].

```
Fig. 10. Horizontal direction mask example
```
Since the zero’s column is oriented horizontally, the above
mask can find edges in that direction. When you apply this

```
mask to an image, you will notice that the image has prominent
horizontal edges.
WORKING OF HORIZONTAL MASK: The horizontal
edges of a picture will be highlighted with this mask. It
operates on the same principle as the above mask, calculating
the difference between the pixel intensities of a specific edge.
Since the mask’s center row is made up of zeros, it ignores
the image’s original edge values and instead measures the
difference between the above and below pixel intensities of
the specific edge. As a result, the abrupt shift in intensities
is amplified, and the edge becomes more noticeable. The
derivative mask concept is used in both of the masks above.
Both masks have the same opposite symbol, and their sum
is zero. The third condition would not apply in this operator
because both of the above masks are standardized and their
values cannot be changed. We will use original image from
[fig 3] to see output of Vertical and Horizontal masks in [fig
11],
```
```
Fig. 11. Vertical and Horizontal Mask Output
```
```
If we combine both the vertical mask and horizontal mask
image, we will get the output of Prewitt Edge detection
algorithm which is shown in [fig 12].
B. SMOOTHING ALGORITHM
Smoothing algorithms are either global or local in that they
take data and filter out noise over a larger, global series or a
smaller, local series by summarizing a local or global domain
```

```
Fig. 12. PREWITT EDGE OUTPUT
```
of Y, resulting in a smooth estimate of the underlying data.
The smoother you can use is determined by the goal of your
study and the peculiarities of your results.
We have implemented few smoothing algorithms for our
analysis purpose which are explained in this project report.

1) GAUSSIAN BLUR ALGORITHM:A Gaussian Filter is
a low-pass filter that is used to distort image regions and
minimize noise (high-frequency components). The low-pass
filter is implemented as an Odd sized symmetric kernel which
passes through each pixel of the area of interest to achieve
it’s desired goal. Since the pixels in the kernel’s middle have
more weight-age towards the final value than those on the
periphery, the kernel is not sensitive to drastic color changes
(edges). A Gaussian Filter can be thought of as a Gaussian
Function approximation.
For example, a 3*3 Gaussian Kernel approximation with
standard deviation = 1 can be calculated using the Gaussian
Blur process, as shown in [fig 13].

```
Fig. 13. Gaussian Blur Output
```
The Gaussian equation, which is given in [fig 14], is used
to compute the values within the kernel:
A Gaussian kernel of any size can be determined using the
function mentioned in [fig 14] by supplying it with estimated
values. We’ll get a performance of the original image below
after we’ve completed all of the necessary steps. The Output
of Gaussian Blur Algorithm is given in [fig 15].

```
Fig. 14. Gaussian Equation
```
```
Fig. 15. Gaussian Blur Output
```
```
2) MEDIAN BLUR ALGORITHM: For blurring images,
the Median Blur algorithm employs a median filter. In most
cases, the median filter is used to minimize image noise.
It’s similar to the mean filter. However, it also does a better
job of maintaining useful information in the picture than the
mean filter. The median filter is a non-linear optical filtering
technique for removing noise from images and signals.
This form of noise reduction is a common pre-processing
phase used to enhance the results of subsequent processing.
Median filtering is commonly used in digital image processing
because it retains edges while suppressing noises under certain
conditions, and it also has applications in signal processing.
We can see the output of Median Blur algorithm in the [fig
16].
```
```
Fig. 16. Median Blur Output
```

One of the key drawbacks of the median filter is that
it is time-consuming and difficult to implement. To find
the median, all of the values in the neighborhood must be
sorted into numerical order, which takes time even with a
fast-sorting algorithm like quick-sort.

3) BILATERAL BLUR ALGORITHM:A bilateral filter is a
non-linear image smoothing filter that preserves edges while
minimizing noise. It uses a weighted average of intensity
values from nearby pixels to replace the intensity of each pixel.
In the case of a bilateral filter, an additional edge term is
needed in addition to the Gaussian blur technique which is
shown in the [fig 17].

```
Fig. 17. Bilateral Equation
```
The output of bilateral blur algorithm can be seen in the
[fig 18].

```
Fig. 18. Bilateral blur algorithm output
```
### V. APPROACHES USED

For the analysis purpose, we have used a total of three
approaches,
1) Single Core Approach
2) Multi-Core Approach
3) GPU Core Approach
Our primary goal is to examine the output time of thousands
of images when run on single-core, multi-core, and GPU cores.
We can learn about the advantages of using a multi-core or
GPU built-in machine as a result of this. Many of us are
unaware of the benefits that these sophisticated systems can
have.
We have used OpenCV library functions to perform image
processing. OpenCV is a large open-source library for com-
puter vision, machine learning, and image processing, and it
now plays a critical role in real-time operations, which are
essential in today’s systems [13].

### A. SINGLE-CORE

```
A single-core processor is a microprocessor with only one
core on a chip and can only run and compute one thread at
a time. Through doing so, the single-core machine absorbs a
significant amount of time. Since the introduction of multi-
core processors, which have several separate processors on a
single chip and can perform several functions simultaneously,
there was a significant change.
The majority of algorithms, or we can say codes that we
write, are run on single-core. Taking an example here, When
we launch the python script, the python binary launches a
Python interpreter who is a Python process. After that, the
operating system assigns the Python program to a single core
until it gets completed. That is all well and good, but we
only utilize a small amount of our true processing power.
Technology Used: OpenCV, pathlib2, numpy, matplotlib,
scipy, imagio
```
```
For the analysis part, we have used three systems mentioned
in the system used section. Moreover, for single-core analysis
purposes, we have implemented mentioned image processing
algorithms on a single core.
B. MULTI-CORE
In Parallel processing, many processes are carried out
simultaneously. Significant problems are divided into smaller
chunks that can be executed simultaneously at the same time.
Parallel processing is the simultaneous processing of the same
task on two or more processor cores to obtain faster results.
A parallel system combines two or more processors (cores)
to solve the problem with shared memory. Parallel processing
can be implemented on a single computer with multiple
processor cores or computers connected by a network or a
combination of both. Parallel processing will utilize multiple
processors and multiple cores in processors using shared
memory to form a shared address space. In parallel process-
ing, multiple computations can be performed simultaneously,
reducing the speed of processing. Therefore, the projects
requiring complex computations and more time mainly use
parallel processing techniques.
It increases the system’s efficiency as it uses all the cores of
the system, which reduces the time of processing. Multi-core
CPU parallelism of image processing algorithms is achieved
by using Python language.
Technology Used: Multiprocessing, OpenCV, pathlib2,
numpy, matplotlib, scipy, imageio, time, OS, argparse, imutils
```
```
1) MULTIPROCESSING LIBRARY:Multiprocessing is the
ability of the system to run several processes at the same
time. An extensive process is divided into chunks of processes
where each process runs independently in a multiprocessing
system. The OS allocates processes to the processors, which
in term improves the performance of the system. A Pool class
is used to submit processes to individual core processors of
the system.
```

### 2) IMPLEMENTATION PROCEDURE:

- Taking dataset and number of cores as input: The dataset
    used in this application is CALTECH-101. The dataset
    consists of 9144 images. Python OS module has functions
    for interactions with the operating system. OS comes
    under Python’s standard utility modules. The operating
    system’s dependent functionality can be used using this
    module. For example, the number of CPUs in the system
    is given by the os.cpucount() method. If CPUs are not
    found, then it returns none as output. Hence, the applica-
    tion computes several cores by method cpucount() and
    takes the dataset and number of cores as input [fig 19].

```
Fig. 19. Dataset and number of core as input
```
- Splitting of Dataset into N equal sized chunks: To split
    Dataset into N (approximately) equal-sized chunks (one
    chunk per core of the processor), the application is
    defined with a chunk generator. It accepts two parameters:
    - l: List of elements (i.e., file paths).
    - n: Number of N-sized chunks to generate.
    Inside function-sized chunks are formed. Hence the ap-
    plication defines chunk generator in parallelhashing.py
    for splitting of the dataset into N equal-sized chunks [fig
    20].

```
Fig. 20. Split Dataset
```
- Create Pool of processes based on number of cores: The
    application distributes the processing of the dataset across
    our system bus. The Pool class creates the Python pro-
    cesses for each core of the system. Hence, the application
    creates a pool of processes based on a number of cores
    [fig 21].

```
Fig. 21. Pool of Processes Creation
```
- Assign chunk to process: To determine the total number
    of images per process, the application divides the number
    of image paths by the number of processes. The chunk
    function creates a list of N equally-sized lists of image
    paths. The application is mapping each of the chunks to
    an independent process [fig 22].

```
Fig. 22. Assign chunk to Process
```
- Number of processes and Reading of images: Each pro-
    cess runs independently of the other process. It reads the
    chunk of processes assigned to it. Furthermore, it reads
    the images in the database for further processing.
- Processing of images dataset: Various image processing
    edge detection and smoothing algorithms are imple-
    mented on different cores to get faster efficient outputs.
3) BENEFITS OF MULTI-CORE:
- Concurrency: Each Processor/Node in a system can per-
    form tasks concurrently.
- Speed of Execution: Speedup is the extent to which more
    hardware can perform the same task in less time than the
    original system.
- Saves Time: With added hardware speedup holds the task
    constant and measures time savings.
- Reduced Response Time: Good Speedup, additional pro-
    cessors and multiple cores reduce System response Time.
- Utilization of Resources: It helps to use the maximum
    potential processing capacity of resources.

```
C. GPU
Many image processing operations iterate through the image
from pixel to pixel, performing calculations based on the
current pixel and eventually writing each computed value
to an output image. Image processing operations can be
performed in parallel for each pixel since the output values
are independent of one another. Graphical Processing Units
(GPUs) are a strong candidate for massively parallel com-
puting implementations. Since GPUs are made for graphical
operations, the hardware for accessing and manipulating pixels
is well-designed.
In this experimental analysis, we have used high-
performance NVIDIA Graphical Processing Units. NVIDIA
provides Compute Unified Device Architecture (CUDA) to
perform NVIDIA GPU-based computing. CUDA is a parallel
computing platform and programming model that enables
dramatic increases in computing performance by harnessing
the power of the graphics processing unit (GPU) [12]. The
systems we used for performing image processing had CUDA-
enabled GPUs.
Implementing image processing algorithms on GPU
involves complications. We have used the OpenCV CUDA
module to handle the intrinsic complexity of the GPU-
based implementation. The OpenCV CUDA module is a
set of classes and functions to utilize CUDA computational
capabilities. It is implemented using NVIDIA CUDA Runtime
API and supports only NVIDIA GPUs. The OpenCV CUDA
module is simple to use and requires no prior knowledge of
CUDA [15].
```

Technology Used: OpenCV, OpenCV CUDA Module, CUDA,
cuDNN, Python

The following steps represent the high-level GPU-based
approach used for Image Processing:

- Read data in Imageh // Imageh is an image in the host
    memory
- Copy the Imageh to Imaged // Imaged is an image in
    the device memory
- Create customized image processing operation with
    OpenCV CUDA module
- Apply image processing operation to Imaged
- Copy Imaged back to Imageh

VI. DATA-SET
The CALTECH-101 dataset is being used to analyze project
reports. Caltech-101 is a set of images of items from 101
different groups. A single entity is assigned to each image.
There are approximately 40 to 800 images in each category,
for a total of about 9000 images. Images come in a variety
of sizes, with average edge lengths ranging from 200 to 300
pixels. As a result, this edition only has image-level marks.
Below is the link for the homepage and where this dataset
can be found,
CALTECH-101 DATASET

VII. SYSTEM USED FOR ANALYSIS
We have used a total of three systems for this project
implementation and analysis. For future references, we are
going to use System 1, System 2, and System 3 names; as for
each system mentioned in [fig 23],
Specification of systems:

```
Fig. 23. Systems used for analysis
```
As we can see, all these systems contain various function-
alities, for example, Intel processor, generation, cache size,
RAM, etc., and Every system has graphical processing unit
(GPU) functionality which also varies from system to system,
as mentioned in the specifications.

VIII. ANALYSIS
This section explains the analysis of the above-mentioned
seven image processing algorithm based on different system
configurations, single-core, multi-core, and GPU core.

### A. SYSTEM 1 ANALYSIS

```
This part provides the analysis of the various image pro-
cessing algorithms done on System 1. We have provided the
table and graph of the time consumed by each algorithm on
System 1. The System 1 configurations are provided in the
earlier section [fig 24] [fig 25].
```
```
Fig. 24. System 1 analysis (time in seconds)
```
```
Fig. 25. System 1 analysis graph
```
### B. SYSTEM 2 ANALYSIS

```
This part provides the analysis of the various image pro-
cessing algorithms done on System 2. We have provided the
table and graph of the time consumed by each algorithm on
System 2. The System 2 configurations are provided in the
earlier section [fig 26] [fig 27].
```
```
Fig. 26. System 2 analysis (time in seconds)
```
### C. SYSTEM 3 ANALYSIS

```
This part provides the analysis of the various image pro-
cessing algorithms done on System 3. We have provided the
```

```
Fig. 27. System 2 analysis graph
```
table and graph of the time consumed by each algorithm on
System 3. The System 3 configurations are provided in the
earlier section [fig 28] [fig 29].

```
Fig. 28. System 3 analysis (time in seconds)
```
```
Fig. 29. System 3 analysis graph
```
From all the graphs mentioned above, we can observe that
a Single-core CPU takes much more time than Multi-core
and GPU. These results show that the time used by image

```
processing algorithms is less when run in parallel with Multi-
core CPU and GPU.
D. MEDIAN BLUR ANALYSIS
It is observed that the algorithm takes a considerable amount
of time on GPU compared to the CPU while performing me-
dian blur with OpenCV. If we keep the kernel size parameter
increasing in the OpenCV function, the GPU takes almost the
same amount of time, but the time taken by the CPU both on
Single-core and Multi-core increases gradually. This section
provides a detailed analysis of the observation mentioned in
figures [fig 30][fig 31][fig 32].
```
```
Fig. 30. System 1 kernel size analysis
```
```
Fig. 31. System 2 kernel size analysis
```
### E. CPU, GPU AND MULTI-CORE ANALYSIS

```
This section provides the overall analysis of the algorithms
on different CPU, GPU, and Multi-Core configurations. We
```

```
Fig. 32. System 3 kernel size analysis
```
have shown time analysis in the figures [fig 33][fig 34][fig
35].

```
Fig. 33. Single core analysis chart (time in seconds)
```
```
Fig. 34. Multi-core analysis chart (time in seconds)
```
```
Fig. 35. GPU analysis chart (time in seconds)
```
After observing the analysis provided in the figure [fig
36][fig 37][fig 38], we can see that i5-9th is a newer generation
processor and the GPU GeForce GTX 1650 has a higher
version than other GPUs used for analysis. Hence, we can see
that the time consumed by these is lesser than other processors
and GPUs.

```
Fig. 36. Single core analysis graph
```
```
Fig. 37. Multi-core analysis graph
```
### IX. CONCLUSION

```
This paper presents multi-core architecture-based image
processing algorithm implementation, which focuses on the
exploitation of parallelism. We have observed that running
image processing algorithms on a vast dataset with multi-core
architecture indeed benefits compared to a single-core CPU.
Furthermore, we achieved good results with good accelerations
for all our OpenCV image processing implementations through
efficient management of multi-core architecture. Our findings
show that complex image processing algorithms requiring
precision and real-time execution can be implemented using
multi-core architecture.
```
```
REFERENCES
[1] “Comparison of Canny edge detector with Sobel and Prewitt edge
detector using different image formats”, Mamta Joshi, Ashutosh Vyas,
IJERT, ISSN 2278-0181.
[2] “A Computational Approach to Edge Detection”, JOHN CANNY, MEM-
BER, IEEE,”IEEE”,VOL.PAMI-8, NO. 6, NOVEMBER 1986.
```

```
Fig. 38. GPU analysis graph
```
[3] C. Nicolescu and P. Jonker. A data and task parallel image processing
environment. Parallel Computing, 28(7-8):945–965, 2002.
[4] “Comparison of Canny edge detector with Sobel and Prewitt edge
detector using different image formats”, Mamta Joshi, Ashutosh
Vyas,”International Journal of Engineering Research Technology
(IJERT)”,sISSN: 2278-0181.
[5] “An Improved Median Filtering Algorithm for Image Noise Reduction”,
Youlian Zhu, Cheng Huang , Physics Procedia 25 ( 2012 ) 609 – 616,
2012
[6] “A Descriptive Algorithm for Sobel Image Edge Detection”, O. R.
Vincent, O. Folorunso, Proceedings of Informing Science IT Education
Conference (InSITE) 2009.
[7] “Laplacian Operator-Based Edge Detectors”, Xin Wang, Member, IEEE,
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE
INTELLIGENCE, VOL. 29, NO. 5, MAY 2007.
[8] “Design and Performance Evaluation of Image Processing Algorithms
on GPUs”, In Kyu Park, Nitin Singhal, Man Hee Lee, Sungdae Cho,
Chris W. Kim, IEEE VOL 2 NO.1 ISSSN 1045-9219 January 2011
[9] “Computer vision algorithms acceleration using graphic processors
NVIDIA CUDA”, Mouna Aafif, Yahia Said, Mohamed Atri, Springer
Science, 3335-
[10] “Implementation of Image Enhancement Algorithms and Recursive Ray
Tracing using CUDA”, Mr. Diptarup Saha, Mr. Karan Darji, Dr. Naren-
dra Patel, Dr. Darshak Thakore, ScienceDirect ELSEVIER Publication,
1877-0509 © 2016
[11] “Performance of Medical Image Processing Algorithm Implemented in
CUDA running on GPU based Machine”, Kalaiselvi Thiruvenkadam,
Sriramakrishnan Padmanaban, Somasundaram Karuppanagounder, Re-
searchGate Publication IJISA, DOI: 10.5815/ijisa.2018.01.
[12] CUDA Toolkit Documentation,
docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
[13] OpenCV Documentation,
opencv-python-tutroals.readthedocs.io/en/latest/index.html
[14] Multiprocessing Documentation,
docs.python.org/3/library/multiprocessing.html
[15] OpenCV CUDA Module,
docs.opencv.org/3.4/d2/dbc/cudaintro.html
[16] T. R. Halfhil, ”Parallel Processing with CUDA”, Microprocessor Report,
Jan 2008.
[17] S.YalamanchiliJ.K.Aggarwal, ”Analysis of a model for parallel image
processing”,Volume 18, Issue 1, 1985
[18] A. Asaduzzaman, A. Martinez and A. Sepehri, ”A time-efficient im-
age processing algorithm for multicore/manycore parallel computing,”
SoutheastCon 2015, 2015, pp. 1-5, doi: 10.1109/SECON.2015.7132924.
[19] Z. Yang, Y. Zhu and Y. Pu, ”Parallel image processing based on
CUDA”, International Conference on Computer Science and Software
Engineering, vol. 3, pp. 198-201, 2008.

```
[20] Monika Hemnani, “Parallel processing techniques for high performance
image processing applications”, Electrical Electronics and Computer
Science (SCEECS) 2016 IEEE Students’ Conference on, pp. 1-4, 2016.
[21] H. B. Prajapati and S. K. Vij,“Analytical study of parallel and distributed
image processing,” 2011 International Conference on Image Information
Processing, 2011, pp. 1-6, doi: 10.1109/ICIIP.2011.6108870.
[22] P. Chalermwat, N. Alexandridis, P. Piamsa-Nga and M. O’Connell, “Par-
allel image processing in heterogeneous computing network systems,”
Proceedings of 3rd IEEE International Conference on Image Processing,
1996, pp. 161-164 vol.2, doi: 10.1109/ICIP.1996.560627.
[23] E.R. Komen. Low-level image processing architectures, 1990.
```

