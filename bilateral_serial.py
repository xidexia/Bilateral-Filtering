import numpy as np
from scipy.spatial import distance
import pylab
import os.path
import time

#from timer import Timer


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

	'''
	stacked = [np.empty_like(image) for i in xrange((2*halo+1)**2)]
	stacked_df = [np.empty_like(image) for i in xrange((2*halo+1)**2)]

	idx=0
	for row in range(-halo, 1+halo):
		for col in range(-halo, 1+halo):
			#stacked_df.append(padded[halo+row:height+halo+row, halo+col:width+halo+col]-image)
			#stacked.append(padded[halo+row:height+halo+row, halo+col:width+halo+col])
			stacked_df[idx] = padded[halo+row:height+halo+row, halo+col:width+halo+col]-image
			stacked[idx] = padded[halo+row:height+halo+row, halo+col:width+halo+col]
			idx+=1
	'''

	print "Finish stack"
	stacked = np.dstack(stacked)

	stacked_df = np.dstack(stacked_df)
	stacked_df = np.exp(-0.5*(stacked_df/sigma)**2)*spatial_gaussian

	print "Finish stacked df"
	stacked *= stacked_df

	print "Calculating out image"
	sum_weight = np.sum(stacked_df, 2)
	out_image = np.sum(stacked, 2)

	out_image /= sum_weight

	del stacked
	del stacked_df
	del sum_weight

	return out_image



if __name__ == '__main__':
	#host_image = np.load('image.npz')['image'].astype(np.float32)[::2, ::2].copy()
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
    





