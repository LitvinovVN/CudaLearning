#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "locale.h"
#include <malloc.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <thread>

using ::std::thread;
using ::std::array;
using ::std::vector;
using ::std::cout;
using ::std::endl;
using ::std::ref;
////////////////////


inline void task1(int& n)
{
    for (int i = 0; i < 10; ++i)
        n = n + 1;
}
////////////////////

inline void threadsTest1() {
    std::cout << std::endl << "----- Тест многопоточности 1 ----" << std::endl;

    static array<int, 100> list{};
    vector<thread> threads;

    threads.reserve(list.size());  // Not needed, an optimization.
    for (int& n : list) {  // Use a range-based for loop, not an explicit counting loop
        threads.emplace_back(task1, ::std::ref(n));
    }
    for (auto& thr : threads) {
        thr.join();
    }

    int total = 0;
    for (int const& n : list) {
        total += n;
    }
    std::cout << total << std::endl;
}