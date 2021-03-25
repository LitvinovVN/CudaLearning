#include "grid3dCreate.cu"
#include "threadsTest1.cu"
#include "addWithCuda.cu"
#include "openMPTests.cpp"
#include "mpiTests.cpp"

int main()
{
    // Включение поддержки кириллицы в консоли
    setlocale(LC_CTYPE, "rus");       
    
    bool isExitCmd = false;
    while (!isExitCmd)
    {        
        std::cout << std::endl << "-----------------------------------" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Режимы работы:" << std::endl;
        std::cout << "1. Сведения о системе" << std::endl;
        std::cout << "2. Сведения о видеоадаптерах" << std::endl;
        std::cout << "3. Тест работоспособности GPU. Суммирование массивов из 5 элементов" << std::endl;
        std::cout << "4. Создание объекта 3d сетки" << std::endl;
        std::cout << "5. Тест многопоточности 1" << std::endl;
        std::cout << "6. OpenMP" << std::endl;
        std::cout << "7. MPI" << std::endl;
        std::cout << "Выберите режим работы: ";
        char selected_mode = 0;
        std::cin >> selected_mode;
        
        switch (selected_mode)
        {
         case '1':
            // Отображение параметров системы
            ShowSystemProperties();
            break;
         case '2':
             // Отображение параметров видеокарты
             ShowVideoadapterProperties();
             break;
         case '3':
             // Тест работоспособности GPU. Суммирование массивов
             addWithCudaStart();
             break;
         case '4':
             // Создание объекта 3d сетки
             grid3dCreate();
             break;
         case '5':
             // Тест многопоточности 1
             threadsTest1();
             break;
         case '6':
             // OpenMP
             openMPTests();
             break;
         case '7':
             // MPI
             mpiTests();
             break;
        default:
            break;
        }

        
        std::cout << std::endl << std::endl << "----------------------------------" << std::endl;
        std::cout << "Для продолжения работы нажмите s. Для завершения x: ";
        std::cin >> selected_mode;
        if (selected_mode == 'x')
            isExitCmd = true;
    }       

    return 0;
}







