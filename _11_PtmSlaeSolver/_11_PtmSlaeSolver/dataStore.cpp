#include <stdio.h>
#include "Dim3d.cpp"
#include <malloc.h>
#include <fstream>

/// <summary>
/// Хранилище данных (линейный массив памяти)
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename Tindx, typename Tdata>
struct DataStore {
	/// <summary>
	/// Размерность массива данных
	/// </summary>
	Dim3d<Tindx> dimensions{};

	/// <summary>
	/// Указатель на массив данных в памяти
	/// </summary>
	Tdata* data = nullptr;

	/// <summary>
	/// Конструктор
	/// </summary>
	/// <param name="dimensions">Размерность</param>
	DataStore(Dim3d<Tindx> dimensions) {
		this->dimensions = dimensions;
		auto dataSize = sizeof(Tdata) * dimensions.N;
		if (!(data = (Tdata*)malloc(dataSize))) {
			printf("Allocation error.\n");
		}		
		printf("Constructor: DataStore created!\n");
	}

	DataStore(char* filePath) {
		std::ifstream inputf(filePath);  // окрываем файл для чтения	
		if (inputf.is_open())
		{
			Tindx x, y, z;
			inputf >> x;
			inputf >> y;
			inputf >> z;
			Dim3d<Tindx> dimensions{ x, y, z };
			this->dimensions = dimensions;
						
			auto dataSize = sizeof(Tdata) * dimensions.N;
			if (!(data = (Tdata*)malloc(dataSize))) {
				printf("Allocation error.\n");
			}
			printf("Constructor: DataStore created!\n");

			for (size_t i = 0; i < dimensions.N; i++)
			{
				inputf >> data[i];
			}

			inputf.close();			
		}
	}

	// Деструктор
	~DataStore()
	{
		free(data);
		data = nullptr;
		printf("Destructor: DataStore deleted!\n");
	}

	/// <summary>
	/// Возвращает линейный индекс по указанному 3d-адресу
	/// </summary>
	/// <param name="x">Координата по оси Ox</param>
	/// <param name="y">Координата по оси Oy</param>
	/// <param name="z">Координата по оси Oz</param>
	/// <returns></returns>
	Tindx GetIndex(Tindx x, Tindx y, Tindx z)
	{
		Tindx result = x + y * dimensions.x + z * dimensions.Nxy;
		return result;
	}

	/// <summary>
	/// Возвращает значение, расположенное в массиве по указанному 3d-адресу
	/// </summary>
	/// <param name="x">Координата по оси Ox</param>
	/// <param name="y">Координата по оси Oy</param>
	/// <param name="z">Координата по оси Oz</param>
	/// <returns></returns>
	Tdata GetValue(Tindx x, Tindx y, Tindx z) {
		Tindx index = GetIndex(x, y, z);
		Tdata result = data[index];
		return result;
	}

	/// <summary>
	/// Инициализирует массив нулями
	/// </summary>
	void InitDataByZeros()
	{
		for (Tindx z = 0; z < dimensions.z; z++)
		{			
			for (Tindx y = 0; y < dimensions.y; y++)
			{				
				for (Tindx x = 0; x < dimensions.x; x++)
				{
					Tindx index = GetIndex(x, y, z);
					data[index] = Tdata{ 0 };
				}
			}
		}
	}

	/// <summary>
	/// Инициализирует массив индексами
	/// </summary>
	void InitDataByIndexes()
	{
		for (Tindx z = 0; z < dimensions.z; z++)
		{
			for (Tindx y = 0; y < dimensions.y; y++)
			{
				for (Tindx x = 0; x < dimensions.x; x++)
				{
					Tindx index = GetIndex(x, y, z);
					data[index] = index;
				}
			}
		}
	}

	void Print() {
		for (Tindx z = 0; z < dimensions.z; z++)
		{
			printf("------------- z = %lld -------------\n", z);
			printf("\t\t");
			for (Tindx x = 0; x < dimensions.x; x++)
			{
				printf("%lld\t\t", x);
			}
			printf("\n");

			for (Tindx y = 0; y < dimensions.y; y++)
			{
				printf("y = %lld:\t", y);
				for (Tindx x = 0; x < dimensions.x; x++)
				{
					printf("%lf\t", data[x + y * dimensions.x + z * dimensions.Nxy]);
				}
				printf("\n");
			}
		}			
	}

	/// <summary>
	/// Сохраняет массив в текстовый файл
	/// </summary>
	/// <param name="filePath">Путь к файлу</param>
	void SaveToFile(char* filePath) {
		std::ofstream outf;     // поток для записи
		outf.open(filePath); // окрываем файл для записи
		if (outf.is_open())
		{
			outf << dimensions.x << std::endl << dimensions.y << std::endl << dimensions.z << std::endl;

			for (size_t i = 0; i < dimensions.N; i++)
			{
				outf << data[i] << std::endl;
			}
			
		}
	}

	/// <summary>
	/// Загружает массив из текстового файла
	/// </summary>
	/// <param name="filePath">Путь к файлу</param>
	static DataStore<size_t, float> LoadFromFile(char* filePath) {
		std::ifstream inputf(filePath);  // окрываем файл для чтения	
		if (inputf.is_open())
		{
			size_t x, y, z;
			inputf >> x;
			inputf >> y;
			inputf >> z;

			DataStore<size_t, float> newDataStore(Dim3d<size_t>(x, y, z));

			for (size_t i = 0; i < newDataStore.dimensions.N; i++)
			{
				inputf >> newDataStore.data[i];
			}

			inputf.close();

			return newDataStore;
		}		
	}
};