import numpy as np
import sys

def verify_matrix_multiplication(filename):
    try:
        with open(filename, 'r') as f:
            data = f.read().strip().split()
        idx = 0
        n = int(data[idx]); idx += 1
        
        A = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(int(data[idx])); idx += 1
            A.append(row)
            
        B = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(int(data[idx])); idx += 1
            B.append(row)
            
        C = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(int(data[idx])); idx += 1
            C.append(row)
        
        A_np = np.array(A)
        B_np = np.array(B)
        C_np = np.array(C)
        C_correct = np.dot(A_np, B_np)
        
        print("Матрица A:")
        print(A_np)
        print("\nМатрица B:")
        print(B_np)
        print("\nРезультат программы:")
        print(C_np)
        print("\nПравильный результат (NumPy):")
        print(C_correct)
        
        if np.array_equal(C_np, C_correct):
            print("\n✓ ВЕРИФИКАЦИЯ УСПЕШНА - результаты совпадают")
            return True
        else:
            print("\n✗ ВЕРИФИКАЦИЯ НЕ УСПЕШНА - результаты отличаются")
            diff = np.abs(C_np - C_correct)
            print(f"Максимальная разница: {np.max(diff)}")
            return False
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_matrix_multiplication(sys.argv[1])
    else:
        print("Использование: python verify.py verification_data.txt")