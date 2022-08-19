
#include <stdlib.h>
#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#define num_gpus 5
#define CUDACHECK(call)                                                   \
    do                                                                    \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (cudaSuccess != err)                                           \
        {                                                                 \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void add_kernel(u_int8_t *a, u_int8_t *b, u_int8_t *c, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
        c[id] = a[id] + b[id];
}

struct MyComm
{
    u_int8_t *send_data[num_gpus];
    // u_int8_t *recv_data[num_gpus];
    pthread_t send_thread[num_gpus];
    u_int8_t *send_buffer[num_gpus];
    cudaStream_t send_stream[num_gpus];
    cudaStream_t add_stream[num_gpus];
    cudaEvent_t send_complete_event[num_gpus];
    cudaEvent_t add_complete_event[num_gpus];
    size_t send_size; // size of send_data, should be multiple of num_gpus
};

cudaError_t send_chunk(struct MyComm *comm, int device, int chunk_id, size_t chunk_size, int step)
{
    CUDACHECK(cudaSetDevice(device));
    if (step != 0)
    {
        CUDACHECK(cudaStreamWaitEvent(comm->send_stream[device], comm->add_complete_event[device], 0));
    }
    printf("Sending chunk %d from device %d-%p to device %d-%p\n", chunk_id, device, comm->send_data[device] + chunk_id * chunk_size, (device + 1) % num_gpus, comm->send_buffer[(device + 1) % num_gpus]);
    CUDACHECK(cudaMemcpyAsync(comm->send_buffer[(device + 1) % num_gpus], comm->send_data[device] + chunk_id * chunk_size, chunk_size, cudaMemcpyDeviceToDevice, comm->send_stream[device]));
    CUDACHECK(cudaEventRecord(comm->send_complete_event[device], comm->send_stream[device]));
    return cudaSuccess;
}

cudaError_t add_chunk(struct MyComm *comm, int device, int chunk_id, size_t chunk_size)
{
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaStreamWaitEvent(comm->add_stream[device], comm->send_complete_event[(device + num_gpus - 1) % num_gpus], 0));
    dim3 block_size(512);
    dim3 grid_size((chunk_size+block_size.x-1) / block_size.x);
    printf("Adding chunk %d on device %d-%p-%p size:%d\n", chunk_id, device, comm->send_buffer[device], comm->send_data[device] + chunk_id * chunk_size, chunk_size);
    add_kernel<<<grid_size, block_size, 0, comm->add_stream[device]>>>((u_int8_t *)comm->send_buffer[device], (u_int8_t *)comm->send_data[device] + chunk_id * chunk_size, (u_int8_t *)comm->send_data[device] + chunk_id * chunk_size, chunk_size);
    CUDACHECK(cudaEventRecord(comm->add_complete_event[device], comm->add_stream[device]));
    return cudaSuccess;
}

cudaError_t reduce_step(MyComm *comm, int step)
{
    int chunk_size = comm->send_size / num_gpus;

    for (int device = 0; device < num_gpus; device++)
    {
        int chunk_id = ((num_gpus - step + device) % num_gpus);
        send_chunk(comm, device, chunk_id, chunk_size, step);
    }
    for (int device = 0; device < num_gpus; device++)
    {
        int chunk_id = ((num_gpus - step - 1 + device) % num_gpus);
        add_chunk(comm, device, chunk_id, chunk_size);
    }
    return cudaGetLastError();
}

cudaError_t ring_reduce(MyComm *comm)
{
    for (int i = 0; i < num_gpus - 1; i++)
    {
        reduce_step(comm, i);
    }
    return cudaSuccess;
}

void comm_init(MyComm *comm)
{
    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&comm->send_data[i], comm->send_size);
        cudaMemset(comm->send_data[i], 1, comm->send_size);
        cudaMalloc(&comm->send_buffer[i], comm->send_size / num_gpus);
        cudaEventCreate(&comm->send_complete_event[i]);
        cudaStreamCreate(&comm->send_stream[i]);
        cudaEventCreate(&comm->add_complete_event[i]);
        cudaStreamCreate(&comm->add_stream[i]);
    }
}

void validate(MyComm *comm)
{
    u_int8_t *data = (u_int8_t *)malloc(comm->send_size);
    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaMemcpy(data, comm->send_data[i], comm->send_size, cudaMemcpyDeviceToHost);
        for(int j = 0; j < comm->send_size; j++)
        {
            printf("%d ", data[j]);
        }
        printf("\n");
    }
}

int main()
{
    MyComm comm;
    comm.send_size = 100;
    comm_init(&comm);
    ring_reduce(&comm);
    validate(&comm);
    return 0;
}
