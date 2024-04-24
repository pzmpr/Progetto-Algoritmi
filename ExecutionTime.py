import time
import random
import math
import matplotlib.pyplot as plt
import QuickSelect
import HeapSelect
import MedianOfMedians

def genera_input(n, maxv):
    a = [0] * n
    for i in range(n):
        a[i] = random.randint(0, maxv)
    return a

def resolution():
    start = time.perf_counter()
    while time.perf_counter() == start:
        pass
    stop = time.perf_counter()
    return stop - start

def benchmark(n, maxv, func, risoluzione, max_rel_error=0.001):
    tmin = risoluzione * ( 1 + ( 1 / max_rel_error ) )
    count = 0
    start = time.perf_counter()
    while (time.perf_counter() - start) < tmin:
        a = genera_input(n, maxv)
        if len(a) > 0:  # Check if array is empty
            k = random.randint(1, n)
            func(a, k)
        count += 1
    duration = time.perf_counter() - start
    return duration/count

# Test
if __name__=="__main__":
    risoluzione = resolution()
    nmin = 100
    nmax = 100000
    iters = 100      # quante volte genera un array di tale dim, migliora la precisione
    base = 2 ** ( (math.log(nmax) - math.log(nmin)) / (iters-1) )

    points = [(None, None, None, None, None)] * iters

    for i in range(iters):
        print(f"\r{i}",end='')
        # n = i * 100 #prof version
        n = int(nmin * (base ** i))
        points[i] = (n, benchmark(n, 100, QuickSelect.randomized_quickselect, risoluzione, 0.001)
                        benchmark(n, 100, QuickSelect.quickselect, risoluzione, 0.001)
                        benchmark(n, 100, HeapSelect.heapselect, risoluzione, 0.001)
                        benchmark(n, 100, MedianOfMedians.median_of_medians_select, risoluzione, 0.001))

# Plot
xs, ys1, ys2, ys3, ys4 = zip(*points)
plt.scatter(xs, ys1)
plt.scatter(xs, ys2)
plt.scatter(xs, ys3)
plt.scatter(xs, ys4)
#plt.plot(dimensioni_input, tempi_medi, marker='o', linestyle='-')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Dimensione dell\'input (n)')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.title('Tempo medio di esecuzione di QuickSort Select')
plt.grid(True)
plt.show() 
plt.close()
