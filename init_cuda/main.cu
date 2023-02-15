#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define N 1024 * 1024

void check(int ret, const char *const func_name, const char *file, const int line_num)
{
    if (ret)
    {
        fprintf(stderr, "error [%s:%d] func_name: %s, code: %d\n", file, line_num, func_name, ret);
    }
}

#define my_check_error(val) check((val), #val, __FILE__, __LINE__)
int init_cuda(int &clock_rate)
{
    int count;
    cudaGetDeviceCount(&count);
    printf("device count: %d\n", count);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("name: %s, major %d, mino: %d, total mem: %ld\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    clock_rate = prop.clockRate;
    return 0;
}

int my_error(int i)
{
    return i % 2;
}

__global__ static void sumOfSquares(int *nums, int *result, clock_t *time)
{
    int sum = 0;
    clock_t start = clock();
    for (int i = 0; i < N; i++)
    {
        sum += nums[i] * nums[i];
    }
    *time = clock() - start;
    printf("time: %ld\n", *time);
    *result = sum;
}

int cal_square_sum(int clock_rate)
{
    int h_data[N];
    for (int i = 0; i < N; i++)
    {
        h_data[i] = i;
    }
    int *h_dev, *result;
    checkCudaErrors(cudaMalloc((void **)&h_dev, sizeof(int) * N));
    checkCudaErrors(cudaMemcpy(h_dev, h_data, sizeof(int) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&result, sizeof(int)));
    clock_t *dev_time;
    cudaMalloc((void **)&dev_time, sizeof(clock_t));
    sumOfSquares<<<1, 2, 2>>>(h_dev, result, dev_time);

    clock_t host_time;
    cudaMemcpy(&host_time, dev_time, sizeof(clock_t), cudaMemcpyDeviceToHost);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum : %d, time: %d\n", sum, host_time/clock_rate);
}

int main()
{   
    int clock_rate = 0;
    init_cuda(clock_rate);
    printf("clock rate is %d\n", clock_rate);
    cal_square_sum(clock_rate);
    return 0;
}