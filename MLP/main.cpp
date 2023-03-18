#include <iostream>
#include <cblas.h>
#include "include/Batch.h"
#include <cstring>

using namespace std;

int main() {
    // Создаем матрицу 3х3 в виде одномерного массива
    double matrix[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Batch b(3, 3);
    memcpy(b[0], matrix, 9 * sizeof(double));
    cout << b[2][2] << endl;

    // Создаем вектор длиной 3
    double vector[3] = {1, 2, 3};
    // Создаем выходной вектор длиной 3
    double result[3] = {0, 0, 0};

    // Выполняем умножение матрицы на вектор
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, b[0], 3, vector, 1, 0.0, result, 1);

    // Выводим результат
    std::cout << "Результат: " << result[0] << ", " << result[1] << ", " << result[2] << std::endl;

    return 0;
}
