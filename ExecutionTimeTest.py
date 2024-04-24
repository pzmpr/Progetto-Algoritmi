import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import QuickSelect
#import HeapSelect
#import MedianOfMedians

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
                        benchmark(n, n, QuickSelect.quickselect, risoluzione, 0.001))
                        #benchmark(n, n, HeapSelect.heapselect, risoluzione, 0.001),
                        #benchmark(n, n, MedianOfMedians.median_of_medians_select, risoluzione, 0.001))

# Plot (line of best fit)
#xs, ys1, ys2, ys3, ys4 = zip(*points)
xs, ys1, ys2 =zip(*points)
nxs=np.array(xs)

plt.subplot(7, 1, 1)
plt.plot(xs, ys1)
plt.title("Tempo di esecuzione Randomized QuickSelect")
# plt.scatter(xs, ys1)
a1, b1 = np.polyfit(nxs, ys1, 1)
plt.plot(xs, a1*nxs+b1)
plt.xlabel('Dimensione dell\'input (n)')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.grid(True)

plt.subplot(7, 1, 3)
plt.title("Tempo di esecuzione QuickSelect")
plt.plot(nxs, ys2)
# plt.scatter(xs, ys2)
a2, b2 = np.polyfit(xs, ys2, 1)
plt.plot(xs, a2*nxs+b2)
plt.xlabel('Dimensione dell\'input (n)')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.grid(True)

'''
plt.subplot(7, 1, 5)
plt.title("Tempo di esecuzione Randomized HeapSelect")
plt.plot(xs, ys3)
# plt.scatter(xs, ys3)
#a3, b3 = np.polyfit(xs, ys3, 1)
#plt.plot(xs, a3*xs+b3)
plt.xlabel('Dimensione dell\'input (n)')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.grid(True)


plt.subplot(7, 1, 7)
plt.title("Tempo di esecuzione Randomized MedianOfMediansSelect")
plt.plot(xs, ys4)
# plt.scatter(xs, ys4)
#a4, b4 = np.polyfit(xs, ys4, 1)
#plt.plot(xs, a4*xs+b4)
plt.xlabel('Dimensione dell\'input (n)')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.grid(True)
'''
plt.show()
plt.close()

# Scala logaritmica
# plt.xscale('log')
# plt.yscale('log')
