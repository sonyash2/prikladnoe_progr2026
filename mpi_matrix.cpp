#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

vector<vector<int>> readMatrix(const string& filename, int& size) {
    ifstream file(filename);
    file >> size;
    vector<vector<int>> matrix(size, vector<int>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i][j];
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    int size = matrix.size();
    file << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

void multiplyMPI(const vector<vector<int>>& A, const vector<vector<int>>& B, 
                 vector<vector<int>>& C, int n, int rank, int size) {
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    // Для процесса 0 - упаковываем матрицу A
    vector<int> flat_A;
    if (rank == 0) {
        flat_A.resize(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat_A[i * n + j] = A[i][j];
    }
    
    // Матрица B для всех процессов
    vector<int> flat_B(n * n);
    if (rank == 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat_B[i * n + j] = B[i][j];
    }
    MPI_Bcast(flat_B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    
    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = (i < remainder) ? rows_per_proc + 1 : rows_per_proc;
        sendcounts[i] = rows * n;
        displs[i] = offset * n;
        offset += rows;
    }
    
    // Локальные строки для текущего процесса
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    vector<int> local_A(local_rows * n);
    
    MPI_Scatterv(rank == 0 ? flat_A.data() : nullptr, 
                 sendcounts.data(), displs.data(), MPI_INT,
                 local_A.data(), local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);
    
    vector<int> local_C(local_rows * n, 0);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * flat_B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
   
    if (rank == 0) {
        flat_A.resize(n * n); // используем как буфер для результата
    }
    
    MPI_Gatherv(local_C.data(), local_rows * n, MPI_INT,
                rank == 0 ? flat_A.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = flat_A[i * n + j];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "MPI MATRIX MULTIPLICATION - LABORATORY WORK 3" << endl;
        cout << "Number of processes: " << size << endl << endl;
    }
    
    // Размеры матриц для тестирования
    int test_sizes[] = {200, 400, 800, 1200, 1600, 2000};
    int num_tests = 6;
    
    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        
        // Создаем матрицы на процессе 0
        vector<vector<int>> A, B, C;
        
        if (rank == 0) {
            A.resize(n, vector<int>(n));
            B.resize(n, vector<int>(n));
            C.resize(n, vector<int>(n, 0));
            
            // Генерация случайных матриц
            srand(time(NULL) + rank);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = rand() % 10 + 1;
                    B[i][j] = rand() % 10 + 1;
                }
            }
            
            cout << "Test " << t+1 << ": Matrix size " << n << "x" << n << endl;
        }
        
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        multiplyMPI(A, B, C, n, rank, size);
        
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            double time_taken = end_time - start_time;
            double ops = 2.0 * n * n * n; // количество операций
            double mflops = ops / time_taken / 1e6;
            
            cout << "  Time: " << time_taken << " seconds" << endl;
            cout << "  Operations: " << ops << endl;
            cout << "  MFLOPS: " << mflops << endl;
            
            string filename = "result_" + to_string(n) + ".txt";
            writeMatrix(filename, C);
            cout << "  Result saved to: " << filename << endl << endl;
        }
    }
    
    if (rank == 0) {
        cout << "EXPERIMENTS COMPLETED" << endl;
    }
    
    MPI_Finalize();
    return 0;
}