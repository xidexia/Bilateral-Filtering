// get pixel
inline float
GETPIX (__global float *in_values, int w, int h, int i, int j){
    if(i<0){
        i=0;
    }
    if(i>=w){
        i=w-1;
    }
    if(j<0){
        j=0;
    }
    if(j>=h){
        j=h-1;
    }
    return in_values[i+w*j];
}


/*  No buffer version */
__kernel void
bilateral_filtering_no_buffer(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer, 
           __local float *spatial, __global __read_only float *spatial_dif,
           int w, int h, float sigma,
           int buf_w, int buf_h,
           const int halo)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((y < h) && (x < w)) { // stay in bounds
        float num = 0;
        float den = 0;
        float pixel = in_values[x+w*y];

        int idx = 0;
        for (int i = -halo; i<=halo; ++i){
            for (int j = -halo; j<=halo; ++j){
                float tmp_p = GETPIX(in_values, w, h, x+i, y+j);
                float dif = tmp_p-pixel;

                //float value = exp(-0.5*(i*i+j*j)/9) * exp(-0.5*(dif*dif)/(sigma*sigma));
                float value = spatial_dif[idx] * exp(-0.5*(dif*dif)/(sigma*sigma));
                num += tmp_p*value;
                den += value;
                ++idx;
            }
        }
        out_values[ x + w*y ] = num/den;
    }

}

/* With buffer version */
__kernel void
bilateral_filtering_buffer(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer, 
           __local float *spatial, __global __read_only float *spatial_dif,
           int w, int h, float sigma,
           int buf_w, int buf_h,
           const int halo)
{

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    const int y_lim = get_global_size(1);


    // Decide whether it is odd number group
    const int group_id_y = get_group_id(1);
    const int group_id_x = get_group_id(0);
    const bool isOdd = (group_id_x%2==1);


    // Write buffer
    const int size = (2*halo+1);

    if (idx_1D<buf_w){
        for (int row = 0; row < buf_h; ++row) {
            buffer[row * buf_w + idx_1D] = \
                GETPIX(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    if (idx_1D<size){
        for (int row=0; row<size; ++row){
            spatial[idx_1D + row*size] = spatial_dif[idx_1D + row*size];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    // Each thread in the valid region (x < w, y < h) should calculate the weighted average
    if ((y < h) && (x < w)) { // stay in bounds
        float num = 0;
        float den = 0;
        float pixel = in_values[x+w*y];

        int idx = 0;
        for (int i = -halo; i<=halo; ++i){
            for (int j = -halo; j<=halo; ++j){
                // get value of neighbourhood
                float tmp_p = buffer[buf_x+i + buf_w*(buf_y+j)];
                float dif = tmp_p-pixel;
                float value = spatial[idx] * exp(-0.5*(dif*dif)/(sigma*sigma));
                num += tmp_p*value;
                den += value;
                ++idx;
            }
        }

        out_values[ x + w*y ] = num/den;
    }

}


/* Buffer version with index trick */
__kernel void
bilateral_filtering_index(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer, int mod,
           __local float *spatial, __global __read_only float *spatial_dif,
           int w, int h, float sigma,
           int buf_w, int buf_h,
           const int halo)
{

    // Global position of output pixel
    const int x = get_global_id(0);
    int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    //const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    //const int local_x = get_local_size(0);
    const int local_y = get_local_size(1);

    const int size = (2*halo+1);

    // Write spatial difference gaussian
    if (idx_1D<size){
        for (int row=0; row<size; ++row){
            spatial[idx_1D + row*size] = spatial_dif[idx_1D + row*size];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterate the column
    for (int base = 0; base < h; base += local_y){
        
        // Write buffer
        if (idx_1D < buf_w){
            // First iteration write buffer
            if (base==0) {
                for (int row = 0; row < buf_h; ++row) {
                    buffer[row * buf_w + idx_1D] = \
                        GETPIX(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
                }
            }
            // Rest of the iterations reuse part of the buffer
            else {
                for (int row = buf_h - local_y; row < buf_h; ++row) {
                    buffer[((buf_corner_y + halo + row) & mod) * buf_w + idx_1D] = GETPIX(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
                    
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        if ((y < h) && (x < w)) {
            float num = 0;
            float den = 0;
            float pixel = in_values[x+w*y];

            int idx = 0;
            for (int i = -halo; i<=halo; ++i){
                for (int j = -halo; j<=halo; ++j){

                    // get value of neighbourhood
                    float tmp_p = buffer[buf_x+i + (( y + halo + j ) & mod) * buf_w];
                    float dif = (tmp_p-pixel);
                    float value = spatial[idx] * exp(-0.5*(dif*dif)/(sigma*sigma));
                    num += tmp_p*value;
                    den += value;
                    ++idx;
                }
            }

            out_values[ x + w*y ] = num/den;
        }

        y += local_y;
        buf_corner_y += local_y;
        barrier(CLK_LOCAL_MEM_FENCE);

    }
}



