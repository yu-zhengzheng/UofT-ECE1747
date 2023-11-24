#include "implementation.h"
#include "stdio.h"
#include <cuda_runtime.h>

void printSubmissionInfo()
{
    // This will be published in the leaderboard on piazza
    // Please modify this field with something interesting
    char nick_name[] = "Team Best";

    // Please fill in your information (for marking purposes only)
    char student_first_name[] = "Colin";
    char student_last_name[] = "Yu";
    char student_student_number[] = "1005252959";

    // Printing out team information
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}

__global__ void cudaKernel(int32_t *d_input, int32_t *d_output, size_t size, int lb) {
    int tid = threadIdx.x + blockIdx.x *1024;
    if (tid>=lb) d_output[tid] = d_input[tid] + d_input[tid-lb];
    else d_output[tid]=d_input[tid];
}


/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
 * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    int32_t *d_buff;
    cudaMalloc((void**)&d_buff, size * sizeof(int32_t));
    cudaMemcpy(d_buff, d_input, size * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    int threadsPerBlock = 1024;//max 1024
    int blocksPerGrid = (size+threadsPerBlock-1)/ threadsPerBlock;
    bool flag=true;
    for (int lb=1;lb<size;lb=lb*2){
        // Launch CUDA kernel
        if (flag){
            cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_buff, d_output, size,lb);
        }else{
            cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_buff, size,lb);
        }
       flag=!flag;
    }
    if(flag) {
        cudaMemcpy(d_output, d_buff, size * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    }
    cudaFree(d_buff);
}
