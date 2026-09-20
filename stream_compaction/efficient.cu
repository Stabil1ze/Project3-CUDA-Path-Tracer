#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        namespace {

            const int BLOCK_SIZE = 64;
            const int FUSED_BLOCK_SIZE = 1024;

            int nextPow2(int n) {
                int m = 1;
                while (m < n) {
                    m <<= 1;
                }
                return m;
            }

        }  // namespace

        // Up-sweep phase of the Blelloch scan
        __global__ void kernEfficientUpSweep(int m, int offset, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int active = m / (2 * offset);
            if (index < active) {
                int idx = (index + 1) * (2 * offset) - 1;
                if (idx < m) {
                    data[idx] += data[idx - offset];
                }
            }
        }

        // Down-sweep phase
        __global__ void kernEfficientDownSweep(int m, int offset, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int active = m / (2 * offset);
            if (index < active) {
                int node0 = (2 * index + 1) * offset - 1;
                int node1 = node0 + offset;
                if (node1 < m) {
                    data[node1] += data[node0];
                    data[node0] = data[node1] - data[node0];
                }
            }
        }

        __global__ void kernEfficientSetLastZero(int m, int *data) {
            if (m > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
                data[m - 1] = 0;
            }
        }

        __global__ void kernEfficientUpSweepFused(int m, int firstOffset, int *data) {
            for (int offset = firstOffset; offset < m; offset <<= 1) {
                int active = m / (2 * offset);
                if (threadIdx.x < active) {
                    int idx = (threadIdx.x + 1) * (2 * offset) - 1;
                    data[idx] += data[idx - offset];
                }
                __syncthreads();
            }
        }

        __global__ void kernEfficientDownSweepFused(int m, int lastOffset, int *data) {
            for (int offset = m / 2; offset >= lastOffset; offset >>= 1) {
                int active = m / (2 * offset);
                if (threadIdx.x < active) {
                    int node0 = (2 * threadIdx.x + 1) * offset - 1;
                    int node1 = node0 + offset;
                    data[node1] += data[node0];
                    data[node0] = data[node1] - data[node0];
                }
                __syncthreads();
            }
        }

        // Runs the work-efficient exclusive scan in place on a device array
        void scanDevice(int m, int *data) {
            if (m <= 0) {
                return;
            }
            if (m == 1) {
                kernEfficientSetLastZero<<<1, 1>>>(m, data);
                return;
            }

            const int fusedFirst = (m > 2 * FUSED_BLOCK_SIZE) ? (m / (2 * FUSED_BLOCK_SIZE)) : 1;

            for (int offset = 1; offset < fusedFirst; offset <<= 1) {
                int active = m / (2 * offset);
                int blocks = (active + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernEfficientUpSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
            }
            if (fusedFirst < m) {
                kernEfficientUpSweepFused<<<1, FUSED_BLOCK_SIZE>>>(m, fusedFirst, data);
            }

            kernEfficientSetLastZero<<<1, 1>>>(m, data);

            if (fusedFirst <= m / 2) {
                kernEfficientDownSweepFused<<<1, FUSED_BLOCK_SIZE>>>(m, fusedFirst, data);
            }
            for (int offset = fusedFirst >> 1; offset > 0; offset >>= 1) {
                int active = m / (2 * offset);
                int blocks = (active + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernEfficientDownSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
            }
        }

        // Performs prefix-sum on idata, storing the result into odata
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int m = nextPow2(n);
            int *devData = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devData), m * sizeof(int));
            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (m > n) {
                cudaMemset(devData + n, 0, (m - n) * sizeof(int));
            }

            scCheckCUDAError("Efficient::scan: cudaMalloc / cudaMemcpy / cudaMemset failed");
            timer().startGpuTimer();
            scanDevice(m, devData);
            timer().endGpuTimer();
            scCheckCUDAError("Efficient::scan: kernel execution failed");

            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
            scCheckCUDAError("Efficient::scan: cudaMemcpy(D2H) failed");
            cudaFree(devData);
        }

        // Performs stream compaction on idata, storing the result into odata
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int m = nextPow2(n);
            const int blockSize = 128;
            const dim3 fullBlocks((n + blockSize - 1) / blockSize);

            int *devIData = nullptr;
            int *devBools = nullptr;
            int *devIndices = nullptr;
            int *devOdata = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devIData), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devBools), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devIndices), m * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devOdata), n * sizeof(int));

            cudaMemcpy(devIData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (m > n) {
                cudaMemset(devIndices + n, 0, (m - n) * sizeof(int));
            }

            scCheckCUDAError("Efficient::compact: cudaMalloc / cudaMemcpy / cudaMemset failed");
            timer().startGpuTimer();

            // Map 1/0 keep/remove values into two buffers
            Common::kernMapToBoolean<<<fullBlocks, blockSize>>>(n, devBools, devIData);
            Common::kernMapToBoolean<<<fullBlocks, blockSize>>>(n, devIndices, devIData);

            scanDevice(m, devIndices);

            Common::kernScatter<<<fullBlocks, blockSize>>>(
                n, devOdata, devIData, devBools, devIndices);

            timer().endGpuTimer();
            scCheckCUDAError("Efficient::compact: kernel execution failed");

            int lastIndex = 0;
            cudaMemcpy(&lastIndex, devIndices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + ((idata[n - 1] != 0) ? 1 : 0);
            cudaMemcpy(odata, devOdata, count * sizeof(int), cudaMemcpyDeviceToHost);
            scCheckCUDAError("Efficient::compact: cudaMemcpy(D2H) failed");

            cudaFree(devIData);
            cudaFree(devBools);
            cudaFree(devIndices);
            cudaFree(devOdata);
            return count;
        }
    }
}
