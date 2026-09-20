#include "common.h"

void scCheckCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        // Maps an array for stream compaction
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n) {
                bools[index] = (idata[index] != 0) ? 1 : 0;
            }
        }

        // Performs scatter on an array
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n && bools[index]) {
                odata[indices[index]] = idata[index];
            }
        }

    }
}
