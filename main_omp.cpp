#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h>

using namespace std;
using namespace chrono;

class MatrixMultiplierOMP {
private:
    vector<vector<int>> A, B, C;
    int n;

public:
    MatrixMultiplierOMP(int size) : n(size) {
        A.resize(n, vector<int>(n));
        B.resize(n, vector<int>(n));
        C.resize(n, vector<int>(n, 0));
    }

    void generateRandomMatrices() {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10 + 1;
                B[i][j] = rand() % 10 + 1;
            }
        }
    }

    bool loadMatricesFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) return false;
        file >> n;
        A.resize(n, vector<int>(n));
        B.resize(n, vector<int>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                file >> A[i][j];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                file >> B[i][j];
        file.close();
        C.resize(n, vector<int>(n, 0));
        return true;
    }

    void saveResultToFile(const string& filename) {
        ofstream file(filename);
        file << n << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                file << C[i][j] << " ";
            file << endl;
        }
        file.close();
    }

    void multiplySequential() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    void multiplyParallelOMP() {
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

    double runSequential() {
        auto start = high_resolution_clock::now();
        multiplySequential();
        auto end = high_resolution_clock::now();
        return duration<double>(end - start).count();
    }

    double runParallel() {
        auto start = high_resolution_clock::now();
        multiplyParallelOMP();
        auto end = high_resolution_clock::now();
        return duration<double>(end - start).count();
    }

    void saveMatricesForVerification() {
        ofstream file("verification_data_omp.txt");
        file << n << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                file << A[i][j] << " ";
            file << endl;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                file << B[i][j] << " ";
            file << endl;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                file << C[i][j] << " ";
            file << endl;
        }
        file.close();
    }
};

int main() {
    cout << "Matrix Multiplication with OpenMP\n";
    cout << "==================================\n";

    int n, threads;
    string filename;
    char choice;

    cout << "Enter matrix size: ";
    cin >> n;
    cout << "Enter number of threads (0 for max): ";
    cin >> threads;

    if (threads > 0) omp_set_num_threads(threads);
    else omp_set_num_threads(omp_get_max_threads());

    MatrixMultiplierOMP mm(n);

    cout << "Load from file? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        cout << "Enter filename: ";
        cin >> filename;
        if (!mm.loadMatricesFromFile(filename)) {
            cout << "Error loading file, generating random matrices...\n";
            mm.generateRandomMatrices();
        }
    }
    else {
        mm.generateRandomMatrices();
    }

    cout << "\nMatrix size: " << n << "x" << n << " (" << n * n << " elements)\n";
    cout << "Operations: " << 2LL * n * n * n << "\n";
    cout << "Threads used: " << omp_get_max_threads() << "\n";

    double time_taken;
    cout << "\nMultiplying...\n";

    if (threads == 1) {
        time_taken = mm.runSequential();
        cout << "Sequential multiplication completed\n";
    }
    else {
        time_taken = mm.runParallel();
        cout << "Parallel multiplication with OpenMP completed\n";
    }

    cout << "Time: " << fixed << setprecision(4) << time_taken << " seconds\n";
    cout << "Performance: " << (2LL * n * n * n / time_taken / 1000000) << " MFLOPS\n";

    cout << "\nSave result to file? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        cout << "Enter filename: ";
        cin >> filename;
        mm.saveResultToFile(filename);
        cout << "Result saved\n";
    }

    cout << "\nVerify? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        mm.saveMatricesForVerification();
        cout << "Data saved to verification_data_omp.txt\n";
        cout << "Run: python verify.py verification_data_omp.txt\n";
    }

    return 0;
}