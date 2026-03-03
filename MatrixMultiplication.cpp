#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <random>
#include <iomanip>
#include <string>
using namespace std;
using namespace chrono;

class MatrixMultiplier {
private:
    vector<vector<int>> A, B, C;
    int n;
    int num_threads;

public:
    MatrixMultiplier(int size, int threads = 1) : n(size), num_threads(threads) {
        A.resize(n, vector<int>(n));
        B.resize(n, vector<int>(n));
        C.resize(n, vector<int>(n, 0));
    }

    void generateRandomMatrices() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 10);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = dis(gen);
                B[i][j] = dis(gen);
            }
        }
    }

    bool loadMatricesFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) return false;
        file >> n;
        A.resize(n, vector<int>(n));
        B.resize(n, vector<int>(n));
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) file >> A[i][j];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) file >> B[i][j];
        file.close();
        C.resize(n, vector<int>(n, 0));
        return true;
    }

    void saveResultToFile(const string& filename) {
        ofstream file(filename);
        file << n << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) file << C[i][j] << " ";
            file << endl;
        }
        file.close();
    }

    void multiplySequential() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0;
                for (int k = 0; k < n; k++) C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    void multiplyParallel(int start_row, int end_row) {
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        }
    }

    double runParallel() {
        vector<thread> threads;
        int rows_per_thread = n / num_threads;
        auto start_time = high_resolution_clock::now();
        for (int t = 0; t < num_threads; t++) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
            threads.emplace_back(&MatrixMultiplier::multiplyParallel, this, start_row, end_row);
        }
        for (auto& th : threads) th.join();
        auto end_time = high_resolution_clock::now();
        return duration<double>(end_time - start_time).count();
    }

    void saveMatricesForVerification() {
        ofstream file("verification_data.txt");
        file << n << endl;
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) file << A[i][j] << " "; file << endl; }
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) file << B[i][j] << " "; file << endl; }
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) file << C[i][j] << " "; file << endl; }
        file.close();
    }

    bool compareWithReference(const string& ref_file) {
        ifstream file(ref_file);
        if (!file.is_open()) return false;
        int ref_n;
        file >> ref_n;
        if (ref_n != n) return false;
        vector<vector<int>> ref_C(n, vector<int>(n));
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) file >> ref_C[i][j];
        file.close();
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (C[i][j] != ref_C[i][j]) return false;
        return true;
    }
};

int main() {
    setlocale(LC_ALL, "Russian");
    cout << "Программа умножения матриц\n";
    cout << "==========================\n";

    int n, threads;
    string filename;
    char choice;

    cout << "Введите размер матрицы: ";
    cin >> n;
    cout << "Введите количество потоков (1 для последовательного): ";
    cin >> threads;

    MatrixMultiplier mm(n, threads);

    cout << "Загрузить из файла? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        cout << "Введите имя файла: ";
        cin >> filename;
        if (!mm.loadMatricesFromFile(filename)) {
            cout << "Ошибка загрузки, генерируем случайные матрицы...\n";
            mm.generateRandomMatrices();
        }
    }
    else {
        mm.generateRandomMatrices();
    }

    cout << "\nРазмер задачи: " << n << "x" << n << " (" << n * n << " элементов)\n";
    cout << "Количество операций: " << 2 * n * n * n << "\n";

    double time_taken;
    cout << "\nВыполнение умножения...\n";

    if (threads == 1) {
        auto start = high_resolution_clock::now();
        mm.multiplySequential();
        auto end = high_resolution_clock::now();
        time_taken = duration<double>(end - start).count();
        cout << "Последовательное умножение завершено\n";
    }
    else {
        time_taken = mm.runParallel();
        cout << "Параллельное умножение с " << threads << " потоками завершено\n";
    }

    cout << "Время выполнения: " << fixed << setprecision(4) << time_taken << " секунд\n";
    cout << "Производительность: " << (2 * n * n * n / time_taken / 1000000) << " MFLOPS\n";

    cout << "\nСохранить результат в файл? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        cout << "Введите имя файла: ";
        cin >> filename;
        mm.saveResultToFile(filename);
        cout << "Результат сохранен в " << filename << "\n";
    }

    cout << "\nВыполнить верификацию? (y/n): ";
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        mm.saveMatricesForVerification();
        cout << "Данные сохранены в verification_data.txt\n";
        cout << "Запустите Python скрипт для проверки:\n";
        cout << "python verify.py verification_data.txt\n";
    }

    cout << "\nПрограмма завершена.\n";
    return 0;
}
