#include <iostream>

/////////// Структура 3D-фрагмент ////////
typedef struct
{
    float* data;   // Указатель на массив с данными
    int* sizes;    // Указатель на массив с кол-вом элементов по x, y, z
    int dimension; // Размерность фрагмента
} grid_fragment;


int grid_fragment_get_num_elements(int* sizes, int dimension)
{
    int num_elements = sizes[0];
    for(int i=1; i<dimension; i++)
        num_elements*=sizes[i];
    return num_elements;
}

int grid_fragment_get_num_elements(grid_fragment* fragment)
{
    return grid_fragment_get_num_elements(fragment->sizes, fragment->dimension);
}

void grid_fragment_data_init(grid_fragment* fragment)
{
    int num_elements = grid_fragment_get_num_elements(fragment->sizes, fragment->dimension);
    for(int i = 0; i < num_elements; i++)
        fragment->data[i] = i;
}

void grid_fragment_construct(grid_fragment* fragment,
    int* sizes,
    int dimension)
{    
    int num_elements = grid_fragment_get_num_elements(sizes, dimension);    
    fragment->data = (float*)malloc(num_elements*sizeof(float));
    fragment->sizes = sizes;
    fragment->dimension = dimension;    
}

void grid_fragment_destruct(grid_fragment* fragment, int size)
{
    free(fragment->data);
    free(fragment->sizes);
}

int grid_fragment_size_bytes(grid_fragment* fragment)
{
    int size = sizeof(grid_fragment)
     + fragment->dimension * sizeof(int)
     + grid_fragment_get_num_elements(fragment->sizes, fragment->dimension) * sizeof(float);
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
    int num_elements = grid_fragment_get_num_elements(fragment->sizes, fragment->dimension);
    printf("num_elements = %d\n", num_elements);

    int fragment_size_bytes = grid_fragment_size_bytes(fragment);
    printf("fragment size in bytes: %d\n", fragment_size_bytes);

    printf("data: ");
    for(int i=0; i<num_elements; i++)
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

__global__ void cuda_fragment_print(int* dev_f3d1_sizes, grid_fragment* fragment){
    int i = getGlobalIndex();
    
    printf("\n-----cuda_fragment_print----\n");
    printf("int* dev_f3d1_sizes = {");
    for(int i = 0; i < 3; i++)
        printf("%d ", dev_f3d1_sizes[i]);
    printf("}\n");


    printf("fragment->dimension = %d\n", fragment->dimension);
    printf("fragment->sizes = {");
    for(int i = 0; i < fragment->dimension; i++)
        printf("%d ", fragment->sizes[i]);
    printf("}\n");

    int num_elements = fragment->sizes[0];
    for(int i=1; i<fragment->dimension; i++)
        num_elements*=fragment->sizes[i];
    printf("num_elements = %d\n", num_elements);

    /*int fragment_size_bytes = grid_fragment_size_bytes(fragment);
    printf("fragment size in bytes: %d\n", fragment_size_bytes);*/

    printf("data: ");
    for(int i=0; i<num_elements; i++)
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

int main()
{
    grid_fragment f3d1;
    int sizes[] = {2,3,4};
    grid_fragment_construct(&f3d1, sizes, 3);
    grid_fragment_data_init(&f3d1);
    grid_fragment_print(&f3d1);        

    // Размер структуры и данных, байт
    int dataSize = grid_fragment_size_bytes(&f3d1);
    printf("dataSize = %d bytes\n", dataSize);
    
    // Выделение памяти в GPU

    int *dev_f3d1_sizes;
    cudaMalloc((void**)&dev_f3d1_sizes, 3 * sizeof(int));
    cudaMemcpy(dev_f3d1_sizes, f3d1.sizes, 3 * sizeof(int), cudaMemcpyHostToDevice);

    float *dev_f3d1_data;
    cudaMalloc((void**)&dev_f3d1_data, grid_fragment_get_num_elements(&f3d1) * sizeof(float));
    cudaMemcpy(dev_f3d1_data, f3d1.data, grid_fragment_get_num_elements(&f3d1) * sizeof(float), cudaMemcpyHostToDevice);
    
    grid_fragment *dto_f3d1 = (grid_fragment *)malloc(sizeof(grid_fragment));
    dto_f3d1->dimension = f3d1.dimension;
    dto_f3d1->sizes = dev_f3d1_sizes;
    dto_f3d1->data = dev_f3d1_data;

    grid_fragment *dev_f3d1;    
    cudaMalloc((void**)&dev_f3d1, sizeof(grid_fragment));    
    cudaMemcpy(dev_f3d1, dto_f3d1, sizeof(grid_fragment), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_f3d1->sizes, dev_f3d1_sizes, sizeof(int*), cudaMemcpyHostToDevice);
    
    cuda_fragment_print<<<1,1>>>(dev_f3d1_sizes, dev_f3d1);
        
    //cudaError_t err1 = cudaMemcpy(a, dev_a, dataSize, cudaMemcpyDeviceToHost);
    //printf(cudaGetErrorString (err1));
        
    //cudaFree(dev_a);
    
    //free(a);
    
    return 0;
}