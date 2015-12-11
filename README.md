#### Harvard CS205 Parallel Programming Final Project - Bilateral Filtering using OpenCL
============================================
### Authors
* Ruitao (Toby) Du \<ruitaodu@g.harvard.edu\>
* Xide Xia \<xidexia@g.harvard.edu\>

### Project Website
http://xidexia.github.io/Bilateral-Filtering

### Video
https://youtu.be/TRVlFCA-YxQ


### Background and Motivation
Smoothing is one of the most commonly used image processing methods. Smoothing an image or a data set is to create an approximating function that attempts to capture important patterns in the data while leaving the noise points - the data points of a signal are modified so individual points are reduced and points that are lower than the adjacent points are increased leading to a smoother signal. There are many kinds of image smoothing algorithms such as Gaussian smoothing, Laplace smoothing, and Bilateral smoothing. A bilateral filter is a nonlinear, edge-preserving and noise-reducing smoothing filter for images. However, for high resolution images, it would take a long time to run.

In this project, our goal is to develop an efficient algorithm for bilateral filtering. In traditional image processing, only one pixel’s value is going to be changed at one time. In our project, we plan to improve efficiency via parallel computing. For example, in the smoothing processing, a predefined filter A is applied to the input image. Traditionally, the center of A is multiplied with the current pixel and the adjacent elements of A are multiplied with the adjacent pixels of the image respectively. With the help of parallel programming in python and OpenCL, we implemented multithreads with local buffers. We also tried some index tricks to further speed up the performance.



### Description
This project explores different parallel implementations of bilateral filtering in OpenCL and compares the performance of them with the serial version in python. Below is four methods we implemented. In these methods, we all precompute the domain kernel first so that we don't need to calculate them multiple times. 

**1. Serial version with NumPy:** <br>
	Calculate the output pixel by pixel. For each pixel, we need to calculate the pixel difference within a certain neighborhood. To speed up the process, we vectorize the calculation by utilizing NumPy.
    
    
**2. OpenCL version without buffer:** <br>
	Calculate the output pixels in blockwise parallel. We use OpenCL with different work group size to parallelize the code. Work group sizes are 8×8, 12×12, 16×16, 20×20. Inside the OpenCl code, it directly read from global memory when we access the neighborhood. 
    
    
**3. OpenCL version with buffer:** <br>
	Calculate the output pixels in blockwise parallel. We use OpenCL with different work group size to parallelize the code. Work group sizes are 8×8, 12×12, 16×16, 20×20. Inside the OpenCL code, first we read in all neighborhood of a work group to the buffer. And then we access the neighborhood by reading from local memory. This way can save time on accessing the global memory. 
    
    
**4. OpenCL version with buffer and index trick:** <br>
	Calculate the output pixels in columnwise parallel. In previous OpenCL methods, we put some pixels to the buffer multiple times. Work group sizes are 16×8, 20×8, 24×8, 28×8, 20×4, 24×4. Instead, we were reusing the buffer by introducing an index trick. Also, to increase the percentage of reused buffer, we set the work group size to be a long thin rectangle: <br>
	![](img/IndexOverlap.png) <br>
	In this way, most of the values in buffer can be reused. 



### Code Instructions

First, download [this file](https://s3.amazonaws.com/Harvard-CS205/HW2/image.npz "image.npz") and save it as image.npz in root directory of this repository.

##### Run serial version of bilateral filtering with NumPy.
```
python bilateral_serial.py
```


For OpenCL version, please make sure ``bilateral.cl`` is in the same folder because all kernal functions are in this file.
##### Run openCL version without buffer
```
python bilateral_without_buffer.py
```

##### Run openCL version with buffer
```
python bilateral_buffer.py
```

##### Run openCL version with index trick
```
python bilateral_index.py
```



### Machine Used
**Apple OpenCL version:** OpenCL 1.2 <br>
**CPU:** 
Intel(R) Core(TM) i7-4770HQ CPU @ 2.20GHz <br>
Maximum clock Frequency: 2200 MHz <br>
Maximum allocable memory size: 4294 MB <br>
Maximum work group size 1024 <br>
**GPU:** 
Iris Pro 
Maximum clock Frequency: 1200 MHz <br>
Maximum allocable memory size: 402 MB <br>
Maximum work group size 512 <br>




### Result

#### Parameters
Spatial σ = halo / 2 <br>
Intensity σ = 50 <br>

#### Sample Image
Bilateral filtering of Cat image with different size of neighborhood 
![](img/cat_halo.png) <br>

Bilateral filtering of Harvard Library image with different size of neighborhood
![](img/library_halo.png) <br>



#### Performance

##### Serial vs OpenCL
![](img/serial.png) <br>

##### Workgroup
**Without buffer: Best work group is 16×16** <br>
![](img/without_buffer.png) <br>


**With buffer: Best work group is 16×16** <br>
![](img/with_buffer.png) <br>


**Buffer with index trick: Best work group is 20×8** <br>
![](img/with_index.png) <br>



##### Best method
**Best method: method 3. The one with buffer only.** <br>
![](img/compare2.png) <br>


### Acknowledge
We thank Ray and all CS205 TFs for providing the wonderful course and all helpful instructions.

### Reference
1. Paris, Sylvain, et al. "A gentle introduction to bilateral filtering and its applications." ACM SIGGRAPH 2007 courses. ACM, 2007.

2. Paris, Sylvain, et al. Bilateral filtering: Theory and applications. Now Publishers Inc, 2009.

3. OpenCL, Khronos. "The open standard for parallel programming of heterogeneous systems." Website. URL http://www. khronos. org/opencl. Symposium on Microarchitecture, MICRO.

4. Tomasi, Carlo, and Roberto Manduchi. "Bilateral filtering for gray and color images." Computer Vision, 1998. Sixth International Conference on. IEEE, 1998.
