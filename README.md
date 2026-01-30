# Assignment 4: Гибридные и распределенные вычисления

## Выполненные задачи
1. [cite_start]**CUDA Sum**: Вычисление суммы массива на GPU через атомарные операции в глобальной памяти[cite: 90].
2. [cite_start]**Scan (Prefix Sum)**: Реализация префиксной суммы на CUDA с использованием Shared Memory[cite: 93].
3. [cite_start]**Hybrid CPU-GPU**: Параллельная обработка массива, где 50% данных считает процессор, а 50% — видеокарта[cite: 96, 97].
4. [cite_start]**MPI Distribution**: Распределенное вычисление суммы на 2, 4 и 8 процессах[cite: 99, 100].

## Запуск
- **CUDA/Hybrid**: `nvcc hybrid_cuda.cu -o hybrid -lgomp`
- **MPI**: `mpicxx mpi_task.cpp -o mpi_task`
- **Запуск MPI**: `mpirun -np 4 ./mpi_task`
