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


    out_image = np.empty_like(input_image)

    gpu_in_image = cl.Buffer(context, cl.mem_flags.READ_ONLY, input_image.size * 4)
    gpu_out_image = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out_image.size * 4)

    # Send to the device, non-blocking
    cl.enqueue_copy(queue, gpu_in_image, input_image, is_blocking=False)

    local_size = (16, 16)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(input_image.shape[::-1], local_size)])
    print "Global size:", global_size
    width = np.int32(input_image.shape[1])
    height = np.int32(input_image.shape[0])

    # sigma for intensity
    sigma = np.float32(50)


    res = []
    for halo_norm in range(1, 11):
        halo = np.int32(halo_norm)
        # Set up a (N+2 x N+2) local memory buffer.
        # +2 for 1-pixel halo on all sides, 4 bytes for float.
        local_memory = cl.LocalMemory(4 * (local_size[0] + 2 * halo) * (local_size[1] + 2 * halo))
        # Each work group will have its own private buffer.
        buf_width = np.int32(local_size[0] + 2 * halo)
        buf_height = np.int32(local_size[1] + 2 * halo)

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
        
        prop_exec = program.bilateral_filtering_buffer(queue, global_size, local_size,
                                                       gpu_in_image, gpu_out_image, local_memory,
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
    pylab.title('before - zoom')
    pylab.figure()

    pylab.imshow(out_image[1200:1800, 3000:3500])
    pylab.title("after - zoom")
    pylab.show()
