#include "libgpu/device.h"
#include <libgpu/context.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "cl/sum_cl.h"
#include "libgpu/shared_device_buffer.h"
#include "libgpu/work_size.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)



int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u a_gpu;
    a_gpu.resizeN(n);
    a_gpu.writeN(as.data(), n);

    unsigned int host_sum = 0;

    gpu::gpu_mem_32u sum_res;
    sum_res.resizeN(1);
    sum_res.writeN(&host_sum, 1);

    const unsigned int workGroupSize = 128;
    const unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    {        
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_global_atomic");
        kernel.compile(true);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            host_sum = 0;
            sum_res.writeN(&host_sum, 1);
            kernel.exec(
                gpu::WorkSize(workGroupSize, globalWorkSize),
                a_gpu, 
                sum_res,
                n
            );
            sum_res.readN(&host_sum, 1);
            EXPECT_THE_SAME(reference_sum, host_sum, "CPU and sum_global_atomic results should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_global_atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_global_atomic: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {        
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_cycle");
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            host_sum = 0;
            sum_res.writeN(&host_sum, 1);
            kernel.exec(
                gpu::WorkSize(workGroupSize, globalWorkSize / 64),
                a_gpu, 
                sum_res,
                n
            );
            sum_res.readN(&host_sum, 1);
            EXPECT_THE_SAME(reference_sum, host_sum, "CPU and sum_cycle results should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_cycle: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {        
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_cycle_coalesce");
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            host_sum = 0;
            sum_res.writeN(&host_sum, 1);
            kernel.exec(
                gpu::WorkSize(workGroupSize, globalWorkSize / 64),
                a_gpu, 
                sum_res,
                n
            );
            sum_res.readN(&host_sum, 1);
            EXPECT_THE_SAME(reference_sum, host_sum, "CPU and sum_cycle_coalesce results should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_cycle_coalesce: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_cycle_coalesce: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {        
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_local_mem");
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            host_sum = 0;
            sum_res.writeN(&host_sum, 1);
            kernel.exec(
                gpu::WorkSize(workGroupSize, globalWorkSize),
                a_gpu, 
                sum_res,
                n
            );
            sum_res.readN(&host_sum, 1);
            EXPECT_THE_SAME(reference_sum, host_sum, "CPU and sum_local_mem results should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_local_mem: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_local_mem: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {        
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            host_sum = 0;
            sum_res.writeN(&host_sum, 1);
            kernel.exec(
                gpu::WorkSize(workGroupSize, globalWorkSize),
                a_gpu, 
                sum_res,
                n
            );
            sum_res.readN(&host_sum, 1);
            EXPECT_THE_SAME(reference_sum, host_sum, "CPU and sum_tree results should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_tree: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
