#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>

using namespace std;

int stringToInt(const string& str) {
    return atoi(str.c_str());
}

vector<vector<int> > readMatrix(const string& filename, int& size) {
    ifstream file(filename.c_str());
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        return vector<vector<int> >();
    }
    
    file >> size;
    vector<vector<int> > matrix(size, vector<int>(size));
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int> >& matrix) {
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error writing to file: " << filename << endl;
        return;
    }
    
    int size = matrix.size();
    file << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

void multiplyMPI(const vector<vector<int> >& A, const vector<vector<int> >& B,
                 vector<vector<int> >& C, int n, int rank, int size) {

    vector<int> flat_A(n * n);
    vector<int> flat_B(n * n);
    vector<int> flat_C(n * n, 0);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_A[i * n + j] = A[i][j];
                flat_B[i * n + j] = B[i][j];
            }
        }
    }
    
    MPI_Bcast(flat_B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    
    vector<int> send_counts(size, 0);
    vector<int> displs(size, 0);
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    for (int i = 0; i < size; i++) {
        send_counts[i] = (i < remainder ? rows_per_proc + 1 : rows_per_proc) * n;
        displs[i] = (i == 0 ? 0 : displs[i-1] + send_counts[i-1]);
    }
    
    int local_rows = (rank < remainder ? rows_per_proc + 1 : rows_per_proc);
    vector<int> local_A(local_rows * n);
    
    MPI_Scatterv(flat_A.data(), send_counts.data(), displs.data(), MPI_INT,
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

    MPI_Gatherv(local_C.data(), local_rows * n, MPI_INT,
                flat_C.data(), send_counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = flat_C[i * n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <matrix_file1> <matrix_file2>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string fileA = argv[1];
    string fileB = argv[2];
    
    int n;
    vector<vector<int> > A, B, C;
    
    if (rank == 0) {
        A = readMatrix(fileA, n);
        B = readMatrix(fileB, n);
        
        if (A.empty() || B.empty()) {
            cerr << "Error reading matrices" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C.resize(n, vector<int>(n, 0));
        
        cout << "Matrices read successfully. Size: " << n << "x" << n << endl;
        cout << "Using " << size << " MPI processes" << endl;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        A.resize(n, vector<int>(n));
        B.resize(n, vector<int>(n));
        C.resize(n, vector<int>(n));
    }
    
    for (int i = 0; i < n; i++) {
        MPI_Bcast(&A[i][0], n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[i][0], n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    double start_time = MPI_Wtime();
    multiplyMPI(A, B, C, n, rank, size);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        double elapsed = end_time - start_time;
        cout << "Multiplication completed in " << elapsed << " seconds" << endl;
        string result_file = "result_" + fileA.substr(fileA.find_last_of("/\\") + 1) + 
                            "_" + fileB.substr(fileB.find_last_of("/\\") + 1);
        writeMatrix(result_file, C);
        cout << "Result written to " << result_file << endl;
    }
    
    MPI_Finalize();
    return 0;
}