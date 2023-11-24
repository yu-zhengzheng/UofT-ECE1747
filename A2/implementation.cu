#include "implementation.h"
#include "stdio.h"
#include <cuda_runtime.h>

void printSubmissionInfo()
{
    char nick_name[] = "Lower the threshold plz";
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

__global__ void K1(int32_t *d_output, int32_t *S, size_t size) {
    int tid = threadIdx.x + blockIdx.x * 5;
    int start=tid*3180+1;
    int end=start+3179;
    if (end>size) end=size;
    for (int i=start;i<end;i++) d_output[i]+=d_output[i-1];
    S[tid]=d_output[end-1];
}

__global__ void Kogge(int32_t *d_input, int32_t *d_output, size_t size, int lb) {
    int tid = threadIdx.x + blockIdx.x * 1024;
    if (tid>=lb) d_output[tid] = d_input[tid] + d_input[tid-lb];
    else d_output[tid]=d_input[tid];
}

__global__ void K3(int32_t *d_output, int32_t *S, size_t size) {
    int tid = threadIdx.x + blockIdx.x * 256;
    int pid =tid/3180-1;
    if (tid<size &&pid>-1) d_output[tid]+=S[pid];
}


void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    //hierarchical
    int threads=size/3180;
    int threads4=threads * 4;
    int32_t *S;
    cudaMalloc((void**)&S, threads4);
    int32_t *S1;//buffer for S array
    cudaMalloc((void**)&S1, threads4);
    int blocksPerGrid = size/15900+1;
    cudaMemcpy(d_output, d_input, size * 4, cudaMemcpyDeviceToDevice);

    //stage 1
    K1<<<blocksPerGrid, 5>>>(d_output, S, size);

    //stage 2
    blocksPerGrid = (threads+1023)/ 1024;
    bool flag=true;
    for (threads4=1;threads4<threads;threads4+=threads4){
        if (flag) Kogge<<<blocksPerGrid, 1024>>>(S, S1, threads, threads4);
        else Kogge<<<blocksPerGrid, 1024>>>(S1, S, threads, threads4);
        flag = !flag;
    }

    //stage 3
    blocksPerGrid = (size+255)/ 256;
    if(flag) K3<<<blocksPerGrid, 256>>>(d_output, S, size);
    else K3<<<blocksPerGrid, 256>>>(d_output, S1, size);
    cudaFree(S1);
    cudaFree(S);
    cudaDeviceSynchronize();
}
