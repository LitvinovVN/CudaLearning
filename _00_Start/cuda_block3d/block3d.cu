#include <iostream>

// Структура "Трехмерный фрагмент"
struct fragment3d
{    
    int nfx;// Количество узлов в фрагменте по оси Ox
    int nfy;// Количество узлов в фрагменте по оси Oy
    int nfz;// Количество узлов в фрагменте по оси Oz

    double* data; // Массив данных фрагмента
};

// Возвращает размер трёхмерного фрагмента в байтах
size_t fragment3d_sizeof(fragment3d* fragment)
{
    size_t size = sizeof(fragment3d);
    size += fragment->nfx * fragment->nfy * fragment->nfz * sizeof(double);
    return size;
}

/// Создаёт в ОЗУ трёхмерный фргмент и возвращает на него указатель
fragment3d* fragment3d_create_ram(int nfx, int nfy, int nfz)
{
    fragment3d* fragment = (fragment3d*)malloc(fragment3d_sizeof(fragment));

    fragment->nfx = nfx;
    fragment->nfy = nfy;
    fragment->nfz = nfz;

    fragment->data = (double*)malloc(nfx*nfy*nfz*sizeof(double));

    return fragment;
}

// Уничтожает трёхмерный фргмент, размещённый в ОЗУ 
void fragment3d_destruct_ram(fragment3d* fragment)
{
    free(fragment->data);
    free(fragment);
}

// Выводит в консоль информацию о трёхмерном фрагменте
void fragment3d_print(fragment3d* fragment)
{
    printf("----- START fragment3d_print -----\n");   

    printf("fragment.nfx = %d\n", fragment->nfx);
    printf("fragment.nfy = %d\n", fragment->nfy);
    printf("fragment.nfz = %d\n", fragment->nfz);

    printf("fragment3d_sizeof() = %d\n", fragment3d_sizeof(fragment));

    printf("fragment.data:\n");
    for(int k=0;k<fragment->nfz;k++)
    {
        printf("k = %d\n", k);
        for(int j=0;j<fragment->nfy;j++)
        {
            for(int i=0;i<fragment->nfx;i++)
            {
                size_t index = i + j * fragment->nfy + k * fragment->nfy * fragment->nfz;
                printf("%lf ", fragment->data[index]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("----- END fragment3d_print -----\n\n");
}

//////////////////////////////////////////////////////////////////////////////

// Структура "Трехмерный блок"
struct block3d
{
    int fxnum;// Количество фрагментов по оси Ox
    int fynum;// Количество фрагментов по оси Oy
    int fznum;// Количество фрагментов по оси Oz

    int nfx;// Количество узлов в одном фрагменте по оси Ox кроме последнего фрагмента
    int nfy;// Количество узлов в одном фрагменте по оси Oy кроме последнего фрагмента
    int nfz;// Количество узлов в одном фрагменте по оси Oz кроме последнего фрагмента

    int nfxl;// Количество узлов в одном фрагменте по оси Ox в последнем фрагменте
    int nfyl;// Количество узлов в одном фрагменте по оси Oy в последнем фрагменте
    int nfzl;// Количество узлов в одном фрагменте по оси Oz в последнем фрагменте
};

// Возвращает размер структуры в байтах
size_t block3d_sizeof(block3d* block)
{
    size_t size = sizeof(block3d);

    return size;
}

// Создаёт в ОЗУ структуру block3d и возвращает на неё указатель
block3d* block3d_create_ram(int fxnum, int fynum, int fznum,
    int nfx, int nfy, int nfz,
    int nfxl, int nfyl, int nfzl)
{
    block3d* block = (block3d*)malloc(block3d_sizeof(block));
    block->fxnum = fxnum;
    block->fynum = fynum;
    block->fznum = fznum;

    block->nfx = nfx;
    block->nfy = nfy;
    block->nfz = nfz;

    block->nfxl = nfxl;
    block->nfyl = nfyl;
    block->nfzl = nfzl;

    return block;
}

// Выводит в консоль сведения о трёхмерном блоке
__host__ __device__ void block3d_print(block3d* block)
{
    printf("----- START block3d_print -----\n");

    printf("block.fxnum = %d\n", block->fxnum);
    printf("block.fynum = %d\n", block->fynum);
    printf("block.fznum = %d\n", block->fznum);

    printf("block.nfx = %d\n", block->nfx);
    printf("block.nfy = %d\n", block->nfy);
    printf("block.nfz = %d\n", block->nfz);

    printf("block.nfxl = %d\n", block->nfxl);
    printf("block.nfyl = %d\n", block->nfyl);
    printf("block.nfzl = %d\n", block->nfzl);

    printf("block3d_sizeof() = %d\n", block3d_sizeof(block));

    printf("----- END block3d_print -----\n\n");
}


__global__ void cuda_block3d_print(){
    printf("Hello World from GPU!\n");    
}

int main() {
    // 1. Создать трёхмерный блок
    block3d* block1 = block3d_create_ram(4,3,2,7,6,5,4,3,2);
    // 2. Создать трёхмерные фрагменты и добавить их в трёхмерный блок

    // 3. Вывести в консоль сведения о трёхмерном блоке
    block3d_print(block1);

    fragment3d* f1 = fragment3d_create_ram(5,6,7);
    fragment3d_print(f1);
    // 4. Скопировать трёхмерный блок в GPU

    // 5. Вывести в консоль сведения о трёхмерном блоке из CUDA-ядра

    cuda_block3d_print<<<1,1>>>(); 
    return 0;
}