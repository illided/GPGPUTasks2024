#include <iostream>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <unistd.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


const int benchmarkingIters = 1;
const int benchmarkingItersCPU = 10;
// const unsigned int max_n = (1 << 24);
const unsigned int max_n = 8;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as)
{
    const unsigned int n = as.size();

    std::vector<unsigned int> bs(n);
    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        for (int i = 0; i < n; ++i) {
            bs[i] = as[i];
            if (i) {
                bs[i] += bs[i-1];
            }
        }
        t.nextLap();
    }

    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

    return bs;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel prefix_sum_up(prefix_sum_kernel, prefix_sum_kernel_length, "pref_sum_we_up");
    prefix_sum_up.compile();

    ocl::Kernel prefix_sum_down(prefix_sum_kernel, prefix_sum_kernel_length, "pref_sum_we_down");
    prefix_sum_down.compile();

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;

	for (unsigned int n = 8; n <= max_n; n *= 4) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

        const std::vector<unsigned int> cpu_reference = computeCPU(as);

// prefix sum
#if 0
        {
            std::vector<unsigned int> res(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                // TODO
                t.restart();
                // TODO
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
#endif

// work-efficient prefix sum
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);
        {
            std::vector<unsigned int> res(n);

            std::cout << "initially" << std::endl;
            for (const auto& el: as) {
                std::cout << el << " ";
            }
            std::cout << std::endl;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                std::cout << "writing data " << std::endl;
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(as.data(), n);

                t.restart();

                int d = 1;
                for (; (1 << d) <= n; d++) {
                    prefix_sum_up.exec(gpu::WorkSize(8, n >> d), as_gpu, bs_gpu, d, n);
                    std::swap(as_gpu, bs_gpu);
                }

                bs_gpu.readN(res.data(), n);

                std::cout << "after up" << std::endl;
                for (const auto& el: res) {
                    std::cout << el << " ";
                }
                std::cout << std::endl;

                d -= 1;
                for (; d > 0; d--) {
                    prefix_sum_down.exec(gpu::WorkSize(8, n >> d), as_gpu, bs_gpu, d, n);
                    std::swap(as_gpu, bs_gpu);
                }

                as_gpu.readN(res.data(), n);

                std::cout << "after down" << std::endl;
                for (const auto& el: res) {
                    std::cout << el << " ";
                }
                std::cout << std::endl;

                t.nextLap();
            }

            as_gpu.readN(res.data(), n);

            std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
	}
}
