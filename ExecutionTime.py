import time
import random
import math
import numpy
import matplotlib.pyplot as plt
import QuickSelect
import HeapSelect
import MedianOfMedians

# genera array di dimensione n con interi compresi nell'intervallo [0, maxv] 
def genera_input(n, maxv):
    a = [0] * n
    for i in range(n):
        a[i] = random.randint(0, maxv)
    return a

# calcola la risoluzione del clock
def resolution():
    start = time.perf_counter()
    while time.perf_counter() == start:
        pass
    stop = time.perf_counter()
    return stop - start

# n dimensione degli array
# maxv massimo valore degli elementi
# func funzione
# risoluzione del clock
# RETURN: tempo medio dell'esecuzione per una singola istanza 
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
        points[i] = (n, benchmark(n, n, QuickSelect.randomized_quickselect, risoluzione, 0.001),
                        benchmark(n, n, QuickSelect.quickselect, risoluzione, 0.001),
                        benchmark(n, n, HeapSelect.heapselect, risoluzione, 0.001),
                        benchmark(n, n, MedianOfMedians.median_of_medians_select, risoluzione, 0.001))

# Plot (line of best fit)
xs, ys1, ys2, ys3, ys4 = zip(*points)

fig1, ax1 = plt.subplot()
fig.subtitle("Tempo di esecuzione Randomized QuickSelect")
ax1.plot(xs, ys1)
# plt.scatter(xs, ys1)
a1, b1 = np.polyfit(xs, ys1, 1)
ax1.plot(xs, a1*xs+b1)
ax1.xlabel('Dimensione dell\'input (n)')
ax1.ylabel('Tempo medio di esecuzione (secondi)')
ax1.grid(True)

fig1, ax1 = plt.subplot()
fig.subtitle("Tempo di esecuzione QuickSelect")
ax2.plot(xs, ys2)
# plt.scatter(xs, ys2)
a2, b2 = np.polyfit(xs, ys2, 1)
ax2.plot(xs, a2*xs+b2)
ax2.xlabel('Dimensione dell\'input (n)')
ax2.ylabel('Tempo medio di esecuzione (secondi)')
ax2.grid(True)

fig1, ax1 = plt.subplot()
fig.subtitle("Tempo di esecuzione Randomized HeapSelect")
ax3.plot(xs, ys3)
# plt.scatter(xs, ys3)
a3, b3 = np.polyfit(xs, ys3, 1)
ax3.plot(xs, a3*xs+b3)
ax3.xlabel('Dimensione dell\'input (n)')
ax3.ylabel('Tempo medio di esecuzione (secondi)')
ax3.grid(True)

fig1, ax1 = plt.subplot()
fig.subtitle("Tempo di esecuzione Randomized MedianOfMediansSelect")
ax4.plot(xs, ys4)
# plt.scatter(xs, ys4)
a4, b4 = np.polyfit(xs, ys4, 1)
ax4.plot(xs, a4*xs+b4)
ax4.xlabel('Dimensione dell\'input (n)')
ax4.ylabel('Tempo medio di esecuzione (secondi)')
ax4.grid(True)

plt.show()
plt.close()

# Scala logaritmica
# plt.xscale('log')
# plt.yscale('log')
