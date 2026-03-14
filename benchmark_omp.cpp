#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

void multiplyOMP(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<int> thread_counts = {1, 2, 4, 8};
    
    cout << "БЕНЧМАРК УМНОЖЕНИЯ МАТРИЦ С OPENMP\n";
    cout << "====================================\n\n";
    
    cout << "Размер;Потоки;Время(с);Операций;MFLOPS;Ускорение\n";
    
    for (int n : sizes) {
        vector<vector<int>> A(n, vector<int>(n));
        vector<vector<int>> B(n, vector<int>(n));
        vector<vector<int>> C(n, vector<int>(n, 0));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10 + 1;
                B[i][j] = rand() % 10 + 1;
            }
        }
        
        double seq_time = 0;
        
        for (int threads : thread_counts) {
            omp_set_num_threads(threads);
            
            double total_time = 0;
            int repeats = (n <= 400) ? 3 : 1;
            
            for (int r = 0; r < repeats; r++) {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        C[i][j] = 0;
                
                auto start = high_resolution_clock::now();
                multiplyOMP(A, B, C, n);
                auto end = high_resolution_clock::now();
                total_time += duration<double>(end - start).count();
            }
            
            double time_taken = total_time / repeats;
            double mflops = (2LL * n * n * n) / time_taken / 1e6;
            
            if (threads == 1) {
                seq_time = time_taken;
                cout << n << ";" << threads << ";" << fixed << setprecision(4) 
                     << time_taken << ";" << (2LL * n * n * n) << ";" 
                     << setprecision(2) << mflops << ";1.00\n";
            } else {
                double speedup = seq_time / time_taken;
                cout << n << ";" << threads << ";" << fixed << setprecision(4) 
                     << time_taken << ";" << (2LL * n * n * n) << ";" 
                     << setprecision(2) << mflops << ";" 
                     << setprecision(2) << speedup << "\n";
            }
        }
    }
    
    return 0;
}