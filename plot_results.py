import matplotlib.pyplot as plt
import numpy as np

sizes = [200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 8]

time_data = {
    200: [0.85, 0.46, 0.28, 0.21],
    400: [6.20, 3.25, 1.82, 1.24],
    800: [48.50, 24.80, 13.20, 8.40],
    1200: [165.30, 84.20, 44.50, 28.10],
    1600: [390.80, 198.50, 104.20, 65.80],
    2000: [765.40, 388.20, 202.50, 127.80]
}

plt.ioff()

print("Сохраняю график 1...")
plt.figure(1, figsize=(10, 6))
for i, t in enumerate(threads):
    times = [time_data[s][i] for s in sizes]
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=f'{t} потоков')
plt.xlabel('Размер матрицы')
plt.ylabel('Время (секунды)')
plt.title('График 1. Время выполнения от размера матрицы')
plt.legend()
plt.grid(True)
plt.savefig('graph1_time_vs_size.png', dpi=150)
plt.close(1)
print("  graph1_time_vs_size.png сохранен")

print("Сохраняю график 2...")
plt.figure(2, figsize=(10, 6))
for s in sizes:
    base_time = time_data[s][0]
    speedup = [base_time / t for t in time_data[s]]
    plt.plot(threads, speedup, 'o-', linewidth=2, markersize=8, label=f'{s}x{s}')
plt.plot(threads, threads, 'k--', linewidth=2, label='Идеальное ускорение')
plt.xlabel('Количество потоков')
plt.ylabel('Ускорение')
plt.title('График 2. Ускорение от числа потоков')
plt.legend()
plt.grid(True)
plt.savefig('graph2_speedup_vs_threads.png', dpi=150)
plt.close(2)
print("  graph2_speedup_vs_threads.png сохранен")

print("Сохраняю график 3...")
plt.figure(3, figsize=(10, 6))
for s in sizes:
    base_time = time_data[s][0]
    speedup = [base_time / t for t in time_data[s]]
    efficiency = [speedup[i] / threads[i] for i in range(len(threads))]
    plt.plot(threads, efficiency, 'o-', linewidth=2, markersize=8, label=f'{s}x{s}')
plt.xlabel('Количество потоков')
plt.ylabel('Эффективность')
plt.title('График 3. Эффективность от числа потоков')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)
plt.savefig('graph3_efficiency_vs_threads.png', dpi=150)
plt.close(3)
print("  graph3_efficiency_vs_threads.png сохранен")

print("Сохраняю график 4...")
plt.figure(4, figsize=(10, 6))
for i, t in enumerate(threads):
    mflops = [(2 * s * s * s / time_data[s][i] / 1000000) for s in sizes]
    plt.plot(sizes, mflops, 'o-', linewidth=2, markersize=8, label=f'{t} потоков')
plt.xlabel('Размер матрицы')
plt.ylabel('Производительность (MFLOPS)')
plt.title('График 4. Производительность от размера матрицы')
plt.legend()
plt.grid(True)
plt.savefig('graph4_mflops_vs_size.png', dpi=150)
plt.close(4)
print("  graph4_mflops_vs_size.png сохранен")

print("Сохраняю график 5...")
plt.figure(5, figsize=(15, 10))
for idx, s in enumerate(sizes):
    plt.subplot(2, 3, idx+1)
    bars = plt.bar(range(len(threads)), time_data[s], color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Потоки')
    plt.ylabel('Время (с)')
    plt.title(f'Матрица {s}x{s}')
    plt.xticks(range(len(threads)), [str(t) for t in threads])
    for bar, val in zip(bars, time_data[s]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('График 5. Сравнение времени для разных размеров')
plt.tight_layout()
plt.savefig('graph5_comparison_bars.png', dpi=150)
plt.close(5)
print("  graph5_comparison_bars.png сохранен")

print("\nВсе графики сохранены в текущую папку:")
print("1. graph1_time_vs_size.png")
print("2. graph2_speedup_vs_threads.png")
print("3. graph3_efficiency_vs_threads.png")
print("4. graph4_mflops_vs_size.png")
print("5. graph5_comparison_bars.png")