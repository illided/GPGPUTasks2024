__kernel void pref_sum_we_up(
    __global const unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1;
    if (idx < 0 || idx >= n) return;

    printf("up d=%d idx=%d pair_idx=%d \n", d, (int)idx, (int)(idx - (1 << (d - 1))));

    bs[idx] = as[idx] + as[idx - (1 << (d - 1))];
}


__kernel void pref_sum_we_down(
    __global const unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1 + (1 << (d - 1));
    if (idx < 0 || idx >= n) return;
    printf("down d=%d idx=%d pair_idx=%d \n", d, (int)idx, (int)(idx - (1 << (d - 1))));
    bs[idx] = as[idx] + as[idx - (1 << (d - 1))];
}