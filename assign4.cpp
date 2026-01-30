#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <omp.h>

// Задание 1: Сумма элементов (Global Memory)
__global__ void sum_kernel(int* d_arr, long long* d_res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd((unsigned long long*)d_res, (unsigned long long)d_arr[idx]);
}

// Задание 2: Префиксная сумма (Shared Memory - упрощенная версия)
__global__ void prefix_sum_kernel(int* d_in, int* d_out, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    if (thid < n) temp[thid] = d_in[thid];
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        int val = 0;
        if (thid >= offset) val = temp[thid - offset];
        __syncthreads();
        temp[thid] += val;
        __syncthreads();
    }
    if (thid < n) d_out[thid] = temp[thid];
}

void task3_hybrid() {
    int n = 1000000;
    std::vector<int> h_arr(n, 1);
    int mid = n / 2;

    // CPU часть
    long long cpu_sum = 0;
    double start_cpu = omp_get_wtime();
    for(int i = 0; i < mid; i++) cpu_sum += h_arr[i];
    double end_cpu = omp_get_wtime();

    // GPU часть
    int *d_arr;
    long long *d_res, h_res_gpu = 0;
    cudaMalloc(&d_arr, (n - mid) * sizeof(int));
    cudaMalloc(&d_res, sizeof(long long));
    cudaMemcpy(d_arr, h_arr.data() + mid, (n - mid) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(long long));

    double start_gpu = omp_get_wtime();
    sum_kernel<<<(n - mid + 255) / 256, 256>>>(d_arr, d_res, n - mid);
    cudaMemcpy(&h_res_gpu, d_res, sizeof(long long), cudaMemcpyDeviceToHost);
    double end_gpu = omp_get_wtime();

    std::cout << "Hybrid Sum: " << cpu_sum + h_res_gpu << std::endl;
    std::cout << "CPU Time: " << end_cpu - start_cpu << "s | GPU Time: " << end_gpu - start_gpu << "s" << std::endl;
}

int main() {
    task3_hybrid();
    return 0;
}
