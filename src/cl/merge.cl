#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

// #line 5


__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    // Get ids
    const unsigned int gid = get_global_id(0);
    const unsigned int block_id = gid / block_size;
    const unsigned int block_local_id = gid % block_size;

    long int l = 0;
    long int r = 0;
    unsigned int write_start = 0;
    bool use_leq = false;
    if (block_id % 2 == 0) {
        l = (block_id + 1) * block_size;
        r = (block_id + 2) * block_size - 1;
        write_start = block_id * block_size;
        use_leq = false;
    } else {
        l = (block_id - 1) * block_size;
        r = block_id * block_size - 1;
        write_start = (block_id - 1) * block_size;
        use_leq = true;
    }

    // Bin search
    const int element = as[gid];
    const long int initial_l = l;

    while (l <= r) {
        long int mid = (l + r) / 2;
        int hyp = as[mid];

        if (hyp == element) {
            if (use_leq) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        } else if (hyp < element) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    long int N = l - initial_l;

    // Write element
    bs[write_start + N + block_local_id] = element;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
