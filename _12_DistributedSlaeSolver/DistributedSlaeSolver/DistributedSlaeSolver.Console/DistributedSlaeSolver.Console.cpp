#include <iostream>
#include <clocale>
#include <map>
using namespace std;

#pragma region Cluster configuration

struct ComputingDevice
{
    virtual string getUid() = 0;
    virtual void print() = 0;
};

struct ComputingDeviceCPU : ComputingDevice {
    string uid;

    ComputingDeviceCPU(string uid) : uid(uid)
    {
            
    }

    string getUid() {
        return uid;
    }

    void print() override {
        cout << "ComputingDeviceCPU ";
        cout << "uid = " << uid;
        cout << endl;
    }
};

/// <summary>
/// Набор вычислительных устройств (ЦПУ, ГПУ и пр.)
/// </summary>
struct ComputingDevices
{
    /// <summary>
    /// Ассоциативный контейнер для хранения вычислительных устройств
    /// </summary>
    map<string, ComputingDevice&> devices;

    /// <summary>
    /// Добавляет вычислительный узел в набор вычислительных узлов
    /// </summary>
    /// <param name="node"></param>
    void addComputingDevice(ComputingDevice& device) {
        device.print();
        string devUid = device.getUid();
        ComputingDevice& deviceRef = device;
        devices.emplace(device);
        //devices[devUid] = deviceRef;
    }

    void print() {
        auto it = devices.begin();
        for (int i = 0; it != devices.end(); it++, i++) {
            cout << i << ") Ключ: " << it->first << endl;
            cout << " - ";
            auto device = &(it->second);
            device->print();
        }
    }
};

/// <summary>
/// Вычислительный узел
/// </summary>
struct ComputingNode {
    string uid{};
    ComputingDevices computingDevices{};

    ComputingNode()
    {
            
    }

    ComputingNode(string uid) : uid(uid)
    {
            
    }

    void print() {
        cout << "Вычислительный узел: " << uid;
        cout << endl;
    }
};

/// <summary>
/// Набор вычислительных узлов
/// </summary>
struct ComputingNodes
{
    /// <summary>
    /// Ассоциативный контейнер для хранения вычислительных узлов
    /// </summary>
    map<string, ComputingNode> nodes;

    /// <summary>
    /// Добавляет вычислительный узел в набор вычислительных узлов
    /// </summary>
    /// <param name="node"></param>
    void addNode(ComputingNode node) {
        nodes[node.uid] = node;
    }

    void print() {
        auto it = nodes.begin();        
        for (int i = 0; it != nodes.end(); it++, i++) {
            cout << i << ") Ключ: " << it->first << endl;
            cout << " - ";
            auto node = it->second;
            node.print();            
        }
    }
};


/// <summary>
/// Вычислительный кластер
/// </summary>
struct Cluster {
#pragma region Constructors
    Cluster(string uid) : uid(uid)
    {
        cout << "Cluster obj constructor invoked for " << uid << endl;
    }
#pragma endregion

#pragma region Destructor
    ~Cluster()
    {
        cout << "Cluster obj destructor invoked for " << uid << endl;
    }
#pragma endregion         

#pragma region Fields
    string uid{};
    ComputingNodes computingNodes{};
#pragma endregion


    void print() {
        cout << "Вычислительный кластер " << uid << endl;
        computingNodes.print();
    }

    /// <summary>
    /// Добавляет вычислительный узел в кластер
    /// </summary>
    /// <param name="node"></param>
    void addNode(ComputingNode node) {
        computingNodes.addNode(node);
    }

};

#pragma endregion




int main()
{
    setlocale(LC_CTYPE, "rus"); // вызов функции настройки локали

    // Создаём вычислительный кластер
    Cluster cluster{ "cluster-1" };
    
    // Создаём вычислительный узел 1
    ComputingNode node1{ "node-1" };
    // Создаём вычислительные устройства и добавляем их к вычислительному узлу
    auto compDev1cpu = ComputingDeviceCPU{ "computingDeviceCPU-1" };
    node1.computingDevices.addComputingDevice(compDev1cpu);

    // Добавляем вычислительный узел 1 в кластер
    cluster.addNode(node1);

    cluster.print();
}
