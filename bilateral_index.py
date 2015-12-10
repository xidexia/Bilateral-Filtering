#pragma OPENCL EXTENSION cl_khr_fp64 : enable
from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab

import scipy.ndimage as im


def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

def filtering(image, halo=5, sigma=2):
    height = image.shape[0]
    width = image.shape[1]
    #out_image = np.empty_like(image)

    # set spatial sigma be half of the neighborhood
    spatial_sigma = min(1, halo/2)
        
    # precompute the spatial gaussian
    spatial_gaussian = []
    for i in range(-halo, halo+1):
        for j in range(-halo, halo+1):
            spatial_gaussian.append(np.exp(-0.5*(i**2+j**2)/(spatial_sigma**2)))

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
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('bilateral.cl').read()).build(options='')

    input_image = np.load('image.npz')['image'].astype(np.float32)
    #input_image = im.imread('img/cat.png').astype(np.float32)
    print "Input image size:", input_image.shape

    # use this input to check correctness of index trick
    '''
    input_image = np.array([[1,1,1,1,1,1,1,1],
                            [2,2,2,2,2,2,2,2], 
                            [3,3,3,3,3,3,3,3],
                            [4,4,4,4,4,4,4,4],
                            [5,5,5,5,5,5,5,5],
                            [6,6,6,6,6,6,6,6],
                            [7,7,7,7,7,7,7,7],
                            [8,8,8,8,8,8,8,8],
                            [9,9,9,9,9,9,9,9],
                            [0,0,0,0,0,0,0,0]]).astype(np.float32)'''
    

    out_image = np.empty_like(input_image)
    out_image_old = np.empty_like(input_image)

    gpu_in_image = cl.Buffer(context, cl.mem_flags.READ_ONLY, input_image.size * 4)
    gpu_out_image = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out_image.size * 4)

    # Send to the device, non-blocking
    cl.enqueue_copy(queue, gpu_in_image, input_image, is_blocking=False)

    # local size
    local_size = (20, 8) 

    # Set second number of global size equal to second number of local size
    # so that we parallelize in columns
    global_size_old = [round_up(g, l) for g, l in zip(input_image.shape[::-1], local_size)]
    global_size_old[1] = local_size[1]
    global_size = tuple(global_size_old)
    print "Global Size", global_size
    width = np.int32(input_image.shape[1])
    height = np.int32(input_image.shape[0])

    # sigma for intensity
    sigma = np.float32(50)

    # explore times of different neighborhood
    res = []
    for halo_norm in range(1, 11):
        halo = np.int32(halo_norm)

        # Set up a 2^N buffer height
        buf_height = 2
        while( buf_height < local_size[1] + 2 * halo):
            buf_height <<= 1

        # Use bitwise mod instead of the normal mod
        mod = np.int32(buf_height-1)


        # Each work group will have its own private buffer.
        buf_width = np.int32(local_size[0] + 2 * halo)
        buf_height = np.int32(buf_height)
        local_memory = cl.LocalMemory(4 * buf_width * buf_height)

        # set spatial sigma be half of the neighborhood
        spatial_sigma = max(1, halo/2)
        
        # precompute the spatial gaussian
        spatial_gaussian = []
        for i in range(-halo, halo+1):
            for j in range(-halo, halo+1):
                spatial_gaussian.append(np.exp(-0.5*(i**2+j**2)/(spatial_sigma**2)))

        spatial_gaussian = np.array(spatial_gaussian).astype(np.float32)

        spatial = cl.Buffer(context, cl.mem_flags.READ_ONLY, spatial_gaussian.size * 4)
        cl.enqueue_copy(queue, spatial, spatial_gaussian, is_blocking=False)
        local_memory_2 = cl.LocalMemory(4 * (2*halo+1) ** 2)

        total_time = 0
        
        prop_exec = program.bilateral_filtering_index(queue, global_size, local_size,
                                                      gpu_in_image, gpu_out_image, local_memory, mod,
                                                      local_memory_2, spatial,
                                                      width, height, sigma,
                                                      buf_width, buf_height, halo)

        prop_exec.wait()
        total_time = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
        
        print "####### HALO={} #######".format(halo)
        print('Finished after {} ms total'.format(total_time))
        res.append(total_time)
    print "Result is", res

    # Show final result
    cl.enqueue_copy(queue, out_image, gpu_out_image, is_blocking=True)

    
    pylab.gray()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom halo = 10')
    pylab.figure()

    pylab.imshow(out_image[1200:1800, 3000:3500])
    pylab.title("after - zoom halo = 10")
    pylab.show()
