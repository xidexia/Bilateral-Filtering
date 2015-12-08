import numpy as np
from scipy.spatial import distance
import pylab
import os.path
import time

import scipy.ndimage as im

def filtering(image, halo=5, sigma=2):
	height = image.shape[0]
	width = image.shape[1]
	#out_image = np.empty_like(image)

	# pre-calculate spatial_gaussian
	spatial_gaussian = []
	for i in range(-halo, halo+1):
		for j in range(-halo, halo+1):
			spatial_gaussian.append(np.exp(-0.5*(i**2+j**2)/(sigma**2)))

	padded = np.pad(image, halo, mode="edge")

	out_image = np.zeros(image.shape)
	weight = np.zeros(image.shape)

	idx=0
	for row in range(-halo, 1+halo):
		for col in range(-halo, 1+halo):
			value = np.exp(-0.5*(padded[halo+row:height+halo+row, halo+col:width+halo+col]-image)**2) \
					*spatial_gaussian[idx]
			out_image += value*padded[halo+row:height+halo+row, halo+col:width+halo+col]
			weight += value
			idx+=1

	out_image /= weight

	return out_image




if __name__ == '__main__':
    #host_image_filtered = np.zeros_like(host_image)

    input_image = np.load('image.npz')['image'].astype(np.float32)#[1200:1800, 3000:3500]
    #print "Size of image is", input_image.shape
    
    pylab.gray()

    #pylab.imshow(input_image)
    #pylab.title('original image')

    #pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    #pylab.imshow(input_image)
    pylab.title('before - zoom')
    
    for halo in range(1, 5):
	    start = time.time()
	    new_image = filtering(input_image, halo)
	    print "####### HALO={} #######".format(halo)
	    print "Time elapse:", time.time()-start
    
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    #pylab.imshow(new_image)
    pylab.title('after - zoom')
    

    pylab.show()
    





