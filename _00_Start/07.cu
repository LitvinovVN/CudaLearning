#include <iostream>

/////////// Структура 3D-фрагмент ////////
typedef struct
{
    int dimension; // Размерность фрагмента (1, 2 или 3-хмерный)
    int data_num;  // Количество элементов в массиве с данными
    float* data;   // Указатель на массив с данными    
    int* sizes;    // Указатель на массив с кол-вом элементов по x, y, z
    
} grid_fragment;

void grid_fragment_data_init(grid_fragment* fragment)
{    
    for(int i = 0; i < fragment->data_num; i++)
        fragment->data[i] = i;
}

void grid_fragment_construct(grid_fragment* fragment,
    int* sizes,
    int dimension)
{    
    int num_elements = sizes[0];
    for(int i=1; i<dimension; i++)
        num_elements*=sizes[i];
    fragment->data_num = num_elements;
    fragment->data = (float*)malloc(num_elements*sizeof(float));
    fragment->sizes = sizes;
    fragment->dimension = dimension;    
}

void grid_fragment_destruct(grid_fragment* fragment)
{
    free(fragment->data);
    free(fragment->sizes);
}

int grid_fragment_size_bytes(grid_fragment* fragment)
{
    int size = sizeof(grid_fragment)
     + fragment->dimension * sizeof(int)
     + fragment->data_num * sizeof(float);
     return size;
} 

void grid_fragment_print(grid_fragment* fragment)
{
    printf("----- grid_fragment_print -----\n");
    printf("fragment->dimension = %d\n", fragment->dimension);
    printf("fragment->sizes = {");
    for(int i = 0; i < fragment->dimension; i++)
        printf("%d ", fragment->sizes[i]);
    printf("}\n");
    
    printf("fragment->data_num = %d\n", fragment->data_num);

    int fragment_size_bytes = grid_fragment_size_bytes(fragment);
    printf("fragment size in bytes: %d\n", fragment_size_bytes);

    printf("data: ");
    for(int i = 0; i < fragment->data_num; i++)
        printf("%g ", fragment->data[i]);
    printf("\n");

    printf("data in tables by layers: \n");
    if(fragment->dimension == 3)
    {
        for(int k = 0; k < fragment->sizes[2]; k++)
        {
            printf("\tz = %d\n", k);
            for(int j = 0; j < fragment->sizes[1]; j++)
            {
                printf("j = %d: ", j);
                for(int i = 0; i < fragment->sizes[0]; i++)
                {
                    int m = i + j * fragment->sizes[0] + k * fragment->sizes[0] * fragment->sizes[1];
                    printf("%g ", fragment->data[m]);
                }
                printf("\n");    
            }
            printf("-----\n");
        }
    }

    printf("-------------------------------\n");
}

//////////////////////////////////////////
/////////// Структура 3D-блок ////////////
//typedef struct
//{
//    grid3d_fragment *fragments;// Указатель на массив фрагментов
//    int num_x;// Кол-во фрагментов по x
//    int num_y;// Кол-во фрагментов по y
//} grid3d_block;
///////////////////////////////////////////

__device__ int getGlobalIndex()
{
    //Индекс текущего блока в гриде
    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.x;
    //Индекс треда внутри текущего блока
    int threadIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
    //глобальный индекс нити
    int i = blockIndex*blockDim.x*blockDim.y*blockDim.z + threadIndex;
    return i;
}

__global__ void cuda_fragment_add_10(grid_fragment* fragment)
{
    for (size_t i = 0; i < fragment->data_num; i++)
    {
        fragment->data[i]+=10;
    }
    
}

__global__ void cuda_fragment_print(grid_fragment* fragment)
{
    int i = getGlobalIndex();
    
    printf("\n-----cuda_fragment_print----\n");
    
    printf("fragment->dimension = %d\n", fragment->dimension);
    printf("fragment->sizes = {");
    for(int i = 0; i < fragment->dimension; i++)
        printf("%d ", fragment->sizes[i]);
    printf("}\n");
    
    printf("fragment->data_num = %d\n", fragment->data_num);

    printf("data: ");
    for(int i=0; i<fragment->data_num; i++)
        printf("%g ", fragment->data[i]);
    printf("\n");

    printf("data in tables by layers: \n");
    if(fragment->dimension == 3)
    {
        for(int k = 0; k < fragment->sizes[2]; k++)
        {
            printf("\tz = %d\n", k);
            for(int j = 0; j < fragment->sizes[1]; j++)
            {
                printf("j = %d: ", j);
                for(int i = 0; i < fragment->sizes[0]; i++)
                {
                    int m = i + j * fragment->sizes[0] + k * fragment->sizes[0] * fragment->sizes[1];
                    printf("%g ", fragment->data[m]);
                }
                printf("\n");    
            }
            printf("-----\n");
        }
    }
}

/// @brief Копирует фрагмент из ОЗУ в GPU
/// @param fragment 
/// @return 
grid_fragment* grid_fragment_copy_host_to_device(grid_fragment* fragment)
{    
    // Указатель на массив размерностей sizes в GPU
    int *dev_f3d1_sizes;
    cudaMalloc((void**)&dev_f3d1_sizes, 3 * sizeof(int));
    cudaMemcpy(dev_f3d1_sizes, fragment->sizes, 3 * sizeof(int), cudaMemcpyHostToDevice);

    // Указатель на массив данных data в GPU
    float *dev_f3d1_data;
    cudaMalloc((void**)&dev_f3d1_data, fragment->data_num * sizeof(float));
    cudaMemcpy(dev_f3d1_data, fragment->data, fragment->data_num * sizeof(float), cudaMemcpyHostToDevice);
    
    // Фрагмент в ОЗУ, подготовленный к копированию в GPU
    grid_fragment *dto_f3d1 = (grid_fragment*)malloc(sizeof(grid_fragment));
    dto_f3d1->dimension = fragment->dimension;
    dto_f3d1->data_num = fragment->data_num;
    dto_f3d1->sizes = dev_f3d1_sizes;
    dto_f3d1->data = dev_f3d1_data;

    grid_fragment *dev_fragment;// Фрагмент в GPU
    cudaMalloc((void**)&dev_fragment, sizeof(grid_fragment));    
    cudaMemcpy(dev_fragment, dto_f3d1, sizeof(grid_fragment), cudaMemcpyHostToDevice);

    return dev_fragment;
}

/// @brief Копирует фрагмент из GPU в ОЗУ
/// @param fragment 
/// @return 
grid_fragment* grid_fragment_copy_device_to_host(grid_fragment* dev_fragment)
{
    // Фрагмент в ОЗУ, подготовленный к копированию из GPU
    grid_fragment *dto_fragment = (grid_fragment*)malloc(sizeof(grid_fragment));
    cudaMemcpy(dto_fragment, dev_fragment, sizeof(grid_fragment), cudaMemcpyDeviceToHost);
        
    // Указатель на массив размерностей sizes в ОЗУ
    int* fragment_sizes = (int*) malloc(dto_fragment->dimension * sizeof(int));     
    cudaMemcpy(fragment_sizes, &dev_fragment->sizes, dto_fragment->dimension * sizeof(int), cudaMemcpyDeviceToHost);
    dto_fragment->sizes = fragment_sizes;
    printf("fragment_sizes[0] = %d\n",fragment_sizes[0]);

    //grid_fragment_print(dto_fragment);       
    // Указатель на массив данных data в ОЗУ
    float *fragment_data = (float*) malloc(dto_fragment->data_num * sizeof(float));    
    cudaMemcpy(fragment_data, &dev_fragment->data, dto_fragment->data_num * sizeof(float), cudaMemcpyDeviceToHost);
    dto_fragment->data = fragment_data;
    
    printf("dto_fragment->sizes[0] = %d\n",dto_fragment->sizes[0]);
    return dto_fragment;
}

int main()
{
    grid_fragment f3d1;
    int sizes[] = {4,3,2};
    grid_fragment_construct(&f3d1, sizes, 3);
    grid_fragment_data_init(&f3d1);
    //grid_fragment_print(&f3d1);        

    // Размер структуры и данных, байт
    int dataSize = grid_fragment_size_bytes(&f3d1);
    printf("dataSize = %d bytes\n", dataSize);

    // Копируем фрагмент из ОЗУ в GPU и сохраняем указатель на объект в GPU    
    grid_fragment* dev_fragment = grid_fragment_copy_host_to_device(&f3d1);
    cuda_fragment_print<<<1,1>>>(dev_fragment);
    cudaDeviceSynchronize();
    cuda_fragment_add_10<<<1,1>>>(dev_fragment);
    cudaDeviceSynchronize();
    cuda_fragment_print<<<1,1>>>(dev_fragment);
    cudaDeviceSynchronize();

    printf("-------------RESULTS------------\n");
    printf("-------fragment_from_gpu--------\n");
    grid_fragment* fragment_from_gpu = grid_fragment_copy_device_to_host(dev_fragment);
    printf("fragment_from_gpu->dimension = %d\n",fragment_from_gpu->dimension);
    printf("fragment_from_gpu->data_num = %d\n",fragment_from_gpu->data_num);
    printf("fragment_from_gpu->sizes[0] = %d\n",fragment_from_gpu->sizes[0]);
    //grid_fragment_print(fragment_from_gpu);      
    //cudaError_t err1 = cudaMemcpy(a, dev_a, dataSize, cudaMemcpyDeviceToHost);
    //printf(cudaGetErrorString (err1));
        
    cudaFree(dev_fragment);
    //cudaFree(dev_f3d1_data);
    //cudaFree(dev_f3d1_sizes);
    
    grid_fragment_destruct(&f3d1);

    
    return 0;
}