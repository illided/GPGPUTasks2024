__kernel void pref_sum_we_up(
    __global unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1;
    if (idx < 0 || idx >= n) return;

    unsigned int value = as[idx] + as[idx - (1 << (d - 1))];
    bs[idx] = value;
    as[idx] = value;
}


__kernel void pref_sum_we_down(
    __global unsigned int* as,
    __global unsigned int* bs,
    const int d,
    const unsigned int n
) {
    long int gid = get_global_id(0);
    long int idx = (gid + 1) * (1 << d) - 1 + (1 << (d - 1));
    if (idx < 0 || idx >= n) return;
    
    unsigned int value = as[idx] + as[idx - (1 << (d - 1))];
    bs[idx] = value;
    as[idx] = value;
}