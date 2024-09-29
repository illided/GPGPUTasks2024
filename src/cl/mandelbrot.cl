#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, int smoothing,
    __global float* results
)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const int xid = get_global_id(0);
    const int yid = get_global_id(1);

    if (yid > height || xid > width) return;

    float x0 = fromX + (xid + 0.5f) * sizeX / width;
    float y0 = fromY + (yid + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    int iter = 0;
    int result = -1;
    for (; iter < iters; iter++) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2 && result == -1) {
            result = iter;
        }
    }

    if (result == -1) {
        result = iters;
    }

    // float result = iter;
    // if (smoothing != 0 && iter != iters) {
    //     result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
    // }

    results[yid * width + xid] = 1.0f * result / iters;
}
