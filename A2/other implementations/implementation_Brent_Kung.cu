#include "implementation.h"
#include "stdio.h"
#include <cuda_runtime.h>

void printSubmissionInfo()
{
    char nick_name[] = "Team Best";
    char student_first_name[] = "Colin";
    char student_last_name[] = "Yu";
    char student_student_number[] = "1005252959";
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}

__global__ void firstStageRest(int32_t *d_output, int32_t dis, int32_t dis2, size_t size) {
    for (; dis < size; dis *= 2) {
        for (int i = dis*2-1; i < size; i+=dis*2) {
            d_output[i] += d_output[i-dis];
        }
    }
}
__global__ void stepFirstStage(int32_t *d_output, int32_t dis, int32_t dis2, size_t size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int dst=tid*dis2-1;
    if (dst<=size)
        d_output[dst] += d_output[dst-dis];
}

__global__ void secondStageRest(int32_t *d_output, int32_t dis, int32_t dis2, size_t size){
    for (; dis > 32768; dis /= 2) {
        for (int i = dis*2-1+dis; i <size; i+=dis*2) {
            d_output[i] += d_output[i-dis];
        }
    }
}
__global__ void stepSecondStage(int32_t *d_output, int32_t dis, int32_t dis2, size_t size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int dst=(tid+1)*dis2-1+dis;
    if (dst<=size) d_output[dst] += d_output[dst-dis];
}

//Brent-Kung prefix sum algorithm
void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    cudaMemcpy(d_output,d_input,size*sizeof(int32_t),cudaMemcpyDeviceToDevice);
    int threadsPerBlock = 1024;
    int blocksPerGrid;
    for (int x=1;x<=32768;x*=2){
        int ops=(size+x-1)/x/2;
        blocksPerGrid = (ops+threadsPerBlock-1)/ threadsPerBlock;
        stepFirstStage<<<blocksPerGrid,threadsPerBlock>>>(d_output,x,x*2,size);
    }
    firstStageRest<<<1,1>>>(d_output,65536,131072,size);
    secondStageRest<<<1,1>>>(d_output,33554432,67108864,size);
    for (int x=32768;x>0;x/=2){
        int ops=(size+x-1)/x/2;
        blocksPerGrid = (ops+threadsPerBlock-1)/ threadsPerBlock;
        stepSecondStage<<<blocksPerGrid,threadsPerBlock>>>(d_output,x,x*2,size);
    }
    cudaDeviceSynchronize();
}
