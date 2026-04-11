#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

__global__ void multiplyMatrixGPU(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void saveResult(const string& filename, const vector<int>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i * n + j] << " ";
        }
        file << endl;
    }
    file.close();
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    int blockSizes[] = {16, 32, 64, 128, 256};
    
    cout << "CUDA MATRIX MULTIPLICATION" << endl;
  
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "GPU: " << prop.name << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    
    for (int n : sizes) {
        cout << "\nMatrix size: " << n << "x" << n << endl;
        cout << "Operations: " << 2LL * n * n * n << endl;
        
        vector<int> A(n * n);
        vector<int> B(n * n);
        vector<int> C(n * n, 0);
        
        for (int i = 0; i < n * n; i++) {
            A[i] = rand() % 10 + 1;
            B[i] = rand() % 10 + 1;
        }
        
        int *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, n * n * sizeof(int));
        cudaMalloc(&d_B, n * n * sizeof(int));
        cudaMalloc(&d_C, n * n * sizeof(int));
        
        cudaMemcpy(d_A, A.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
        
        for (int bs : blockSizes) {
            dim3 threadsPerBlock(bs, bs);
            dim3 blocksPerGrid((n + bs - 1) / bs, (n + bs - 1) / bs);
            
            cudaMemset(d_C, 0, n * n * sizeof(int));
            
            auto start = high_resolution_clock::now();
            
            multiplyMatrixGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
            cudaDeviceSynchronize();
            
            auto end = high_resolution_clock::now();
            
            cudaMemcpy(C.data(), d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);
            
            double time_taken = duration<double>(end - start).count();
            double ops = 2.0 * n * n * n;
            double mflops = ops / time_taken / 1e6;
            
            cout << "Block: " << bs << "x" << bs 
                 << " | Grid: " << blocksPerGrid.x << "x" << blocksPerGrid.y
                 << " | Time: " << time_taken << " s"
                 << " | MFLOPS: " << mflops << endl;
        }
        
        string filename = "cuda_result_" + to_string(n) + ".txt";
        saveResult(filename, C, n);
        cout << "Saved: " << filename << endl;
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    cout << "DONE!" << endl;
    
    return 0;
}