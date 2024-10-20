__kernel void bitonic(__global int *as, long int red_block_size, long int colour_block_size)
{
    long int gid = get_global_id(0);
    long int half_red_block_size = red_block_size / 2;
    long int half_red_block_id = gid / half_red_block_size;
    long int half_red_inner_id = gid % half_red_block_size;

    long int idx = half_red_block_id * red_block_size + half_red_inner_id;
    long int pair_idx = idx + half_red_block_size;

    bool ascending = (idx / colour_block_size) % 2 == 0;

    int elem1 = as[idx];
    int elem2 = as[pair_idx];

    if ((ascending && (elem1 > elem2)) || (!ascending && (elem1 < elem2))) {
        as[pair_idx] = elem1;
        as[idx] = elem2;
    }
}
