import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(15000)

# ^ QUICKSELECT ^ #

# a array
# p indice di inizio array
# r indice di fine array
# i indice dell'elemento da trovare
def quickselect(a, i):
  return select(a, 0, len(a)-1, i)

def select(a, p, r, i):
  if p == r:
    return a[p]
  else:
    q = partition(a, p, r)
    k = q - p + 1
    if i == k:
      return a[q]
    elif i < k:
      return select(a, p, q-1, i)
    else:
      return select(a, q+1, r, i-k)

# a array di interi
# p inizio dell'array
# r fine dell'array
# esegue partition sull'intervallo [p,r] dell'array a
def partition(a, low, high):
  p = a[high]
  i = low - 1
  for j in range(low, high):
    if a[j] <= p:
      i = i + 1
      a[i], a[j] = a[j], a[i]
  a[i+1], a[high] = a[high], a[i+1] 
  return i + 1

def randomized_quickselect(a, i):
  return randomized_select(a, 0, len(a)-1, i)

def randomized_select(a, p, r, i):
  if p == r:
    return a[p]
  else:
    q = randomized_partition(a, p, r)
    k = q - p + 1
    if i == k:
      return a[q]
    elif i < k:
      return randomized_select(a, p, q-1, i)
    else:
      return randomized_select(a, q+1, r, i-k)

def randomized_partition(a, low, high):
  i = random.randint(low, high-1)
  a[high-1], a[i] = a[i], a[high-1]
  return partition(a, low, high)


# ^ HEAPSELECT ^ #

# Algoritmo HeapSelect con complessita'
# Th(n) sia nel caso pessimo che nel caso medio

# Variante dell'heapselect
# in base al valore di k viene scelta la procedura con minheap o maxheap
def minmax_heapselect(a, k):
    if k > len(a)//2:
        return max_heapselect(a, k)
    else:
        return min_heapselect(a, k)

# a array
# p indice di inizio array
# r indice di fine array
# k indice dell'elemento da trovare
def min_heapselect(a, k):
  main_heap = Minheap()
  main_heap.buildheap(a)
  aux_heap = MinheapAux()
  aux_heap.insert(main_heap.heap[0], 0)

  for i in range(0, k-1):
      (x, j) = aux_heap.getmin()
      aux_heap.extract()

      l = main_heap.left(j)
      r = main_heap.right(j)
      if l != None:
          aux_heap.insert(main_heap.heap[l], l)
      if r != None:
          aux_heap.insert(main_heap.heap[r], r)

  (x, j) = aux_heap.getmin()
  return x


# Implementazione MinHeap
class Minheap:

  # heap: array di interi che rappresentano i valori dei nodi
  def __init__(self):
      self.heap = []

  # RETURN: lunghezza dell'array
  def len(self):
      return len(self.heap)

  # RETURN: valore minimo (radice)
  def getmin(self):
      assert len(self.heap) > 0
      return self.heap[0]

  # i indice di un nodo
  # RETURN: indice del nodo padre
  # se il nodo in posizione i e' la radice, non fa nulla
  def parent(self, i):
      if i == 0:
          return None
      return (i + 1) // 2 - 1

  # i indice di un nodo
  # RETURN: indice del figlio sinistro
  def left(self, i):
      j = i * 2 + 1
      if j >= len(self.heap):
          return None
      return j

  # i indice di un nodo
  # REUTRN: indice del figlio destro
  def right(self, i):
      j = i * 2 + 2
      if j >= len(self.heap):
          return None
      return j    

  # estrae il nodo radice dalla heap
  def extract(self):
      self.heap[0] = self.heap[-1]
      self.heap = self.heap[:-1]
      self.heapify(0)

  # x nodo da inserire
  # inserisce il nodo nella heap
  def insert(self, x):
      self.heap.append(x)
      self.moveup(len(self.heap) - 1)

  # i indice di un nodo della heap
  # x nodo da scambiare
  # scambia il nodo in posizione i nella heap con il nodo x
  def change(self, i, x):
      assert i >= 0 and i < len(self.heap)
      if x < self.heap[i]:
          self.heap[i] = x
          self.moveup(i)
      elif x > self.heap[i]:
          self.heap[i] = x
          self.heapify(i)

  # a array di interi
  # costruisce una minheap dall'array a 
  def buildheap(self, a):
      self.heap = a.copy()
      for i in range(len(self.heap) - 1, -1, -1):
          self.heapify(i)

  # i indice del nodo da cui parte la procedura
  # "sistema" l'albero in modo che il sottoalbero con radice il
  # nodo in posizione i sia una minheap
  def heapify(self, i):
      l = self.left(i)
      r = self.right(i)
      argmin = i
      if l != None and self.heap[l] < self.heap[argmin]:
          argmin = l
      if r != None and self.heap[r] < self.heap[argmin]:
          argmin = r
      if i != argmin:
          self.heap[i], self.heap[argmin] = self.heap[argmin], self.heap[i]
          self.heapify(argmin)

  # i indice di un nodo
  # scambia il nodo in posizione i nella heap con il suo genitore
  # se il nodo in posizione i e' la radice, non fa nulla
  def moveup(self, i):
      if i == 0:
          return
      p = self.parent(i)
      if p != None and self.heap[i] < self.heap[p]:
          self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
          self.moveup(p)

# Implementazione MinHeap Ausiliaria
class MinheapAux:

  #   heap: array di interi che rappresentano i valori dei nodi
  #   ??? pos: array di interi che rappresentano le posizioni dei nodi
  def __init__(self):
      self.heap = []
      self.pos = []

  # RETURN: lunghezza dell'array
  def len(self):
      return len(self.heap)

  # RETURN: valore minimo (radice)
  def getmin(self):
      assert len(self.heap) > 0
      return (self.heap[0], self.pos[0])

  # i indice di un nodo
  # RETURN: indice del nodo padre
  # se il nodo in posizione i e' la radice, non fa nulla
  def parent(self, i):
      if i == 0:
          return None
      return (i + 1) // 2 - 1

  # i indice di un nodo
  # RETURN: indice del figlio sinistro
  def left(self, i):
      j = i * 2 + 1
      if j >= len(self.heap):
          return None
      return j

  # i indice di un nodo
  # REUTRN: indice del figlio destro
  def right(self, i):
      j = i * 2 + 2
      if j >= len(self.heap):
          return None
      return j    

  # estrae il nodo radice dalla heap
  def extract(self):
      self.heap[0] = self.heap[-1]
      self.pos[0] = self.pos[-1]
      self.heap = self.heap[:-1]
      self.pos = self.pos[:-1]
      self.heapify(0)

  # x nodo da inserire
  # inserisce il nodo nella heap
  def insert(self, x, p):
      self.heap.append(x)
      self.pos.append(p)
      self.moveup(len(self.heap) - 1)

  # i indice di un nodo della heap
  # x nodo da scambiare
  # scambia il nodo in posizione i nella heap con il nodo x
  def change(self, i, x):
      assert i >= 0 and i < len(self.heap)
      if x < self.heap[i]:
          self.heap[i] = x
          self.moveup(i)
      elif x > self.heap[i]:
          self.heap[i] = x
          self.heapify(i)

  # a array di interi
  # costruisce una minheap dall'array a 
  def buildheap(self, a):
      self.heap = a.copy()
      for i in range(len(self.heap) - 1, -1, -1):
          self.heapify(i)

  # i indice del nodo da cui parte la procedura
  # "sistema" l'albero in modo che il sottoalbero con randice il
  # nodo in posizione i sia una minheap
  def heapify(self, i):
      l = self.left(i)
      r = self.right(i)
      argmin = i
      if l != None and self.heap[l] < self.heap[argmin]:
          argmin = l
      if r != None and self.heap[r] < self.heap[argmin]:
          argmin = r
      if i != argmin:
          self.heap[i], self.heap[argmin] = self.heap[argmin], self.heap[i]
          self.pos[i], self.pos[argmin] = self.pos[argmin], self.pos[i]
          self.heapify(argmin)

  # i indice di un nodo
  # scambia il nodo in posizione i nella heap con il suo genitore
  # se il nodo in posizione i e' la radice, non fa nulla
  def moveup(self, i):
      if i == 0:
          return
      p = self.parent(i)
      if p != None and self.heap[i] < self.heap[p]:
          self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
          self.pos[i], self.pos[p] = self.pos[p], self.pos[i]
          self.moveup(p)

# Algoritmo HeapSelect con complessita'
# Th(n) sia nel caso pessimo che nel caso medio

# a array
# p indice di inizio array
# r indice di fine array
# k indice dell'elemento da trovare

def max_heapselect(a, k):
  main_heap = Maxheap()
  main_heap.buildheap(a)
  aux_heap = MaxheapAux()
  aux_heap.insert(main_heap.heap[0], 0)

  for i in range(len(a)-1, k-1, -1):
      (x, j) = aux_heap.getmax()
      aux_heap.extract()

      l = main_heap.left(j)
      r = main_heap.right(j)
      if l != None:
          aux_heap.insert(main_heap.heap[l], l)
      if r != None:
          aux_heap.insert(main_heap.heap[r], r)

  (x, j) = aux_heap.getmax()
  return x


# Implementazione MinHeap
class Maxheap:

  # heap: array di interi che rappresentano i valori dei nodi
  def __init__(self):
      self.heap = []

  # RETURN: lunghezza dell'array
  def len(self):
      return len(self.heap)

  # RETURN: valore massimo (radice)
  def getmax(self):
      assert len(self.heap) > 0
      return self.heap[0]

  # i indice di un nodo
  # RETURN: indice del nodo padre
  # se il nodo in posizione i e' la radice, non fa nulla
  def parent(self, i):
      if i == 0:
          return None
      return (i + 1) // 2 - 1

  # i indice di un nodo
  # RETURN: indice del figlio sinistro
  def left(self, i):
      j = i * 2 + 1
      if j >= len(self.heap):
          return None
      return j

  # i indice di un nodo
  # REUTRN: indice del figlio destro
  def right(self, i):
      j = i * 2 + 2
      if j >= len(self.heap):
          return None
      return j    

  # estrae il nodo radice dalla heap
  def extract(self):
      self.heap[0] = self.heap[-1]
      self.heap = self.heap[:-1]
      self.heapify(0)

  # x nodo da inserire
  # inserisce il nodo nella heap
  def insert(self, x):
      self.heap.append(x)
      self.moveup(len(self.heap) - 1)

  # i indice di un nodo della heap
  # x nodo da scambiare
  # scambia il nodo in posizione i nella heap con il nodo x
  def change(self, i, x):
      assert i >= 0 and i < len(self.heap)
      if x > self.heap[i]:
          self.heap[i] = x
          self.moveup(i)
      elif x < self.heap[i]:
          self.heap[i] = x
          self.heapify(i)

  # a array di interi
  # costruisce una maxheap dall'array a 
  def buildheap(self, a):
      self.heap = a.copy()
      for i in range(len(self.heap) - 1, -1, -1):
          self.heapify(i)

  # i indice del nodo da cui parte la procedura
  # "sistema" l'albero in modo che il sottoalbero con radice il
  # nodo in posizione i sia una maxheap
  def heapify(self, i):
      l = self.left(i)
      r = self.right(i)
      argmax = i
      if l != None and self.heap[l] > self.heap[argmax]:
          argmax = l
      if r != None and self.heap[r] > self.heap[argmax]:
          argmax = r
      if i != argmax:
          self.heap[i], self.heap[argmax] = self.heap[argmax], self.heap[i]
          self.heapify(argmax)

  # i indice di un nodo
  # scambia il nodo in posizione i nella heap con il suo genitore
  # se il nodo in posizione i e' la radice, non fa nulla
  def moveup(self, i):
      if i == 0:
          return
      p = self.parent(i)
      if p != None and self.heap[i] > self.heap[p]:
          self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
          self.moveup(p)

# Implementazione MinHeap Ausiliaria
class MaxheapAux:

  # heap: array di interi che rappresentano i valori dei nodi
  # pos: array di interi che rappresentano le posizioni dei nodi nella heap principale
  def __init__(self):
      self.heap = []
      self.pos = []

  # RETURN: lunghezza dell'array
  def len(self):
      return len(self.heap)

  # RETURN: valore massimo (radice)
  def getmax(self):
      assert len(self.heap) > 0
      return (self.heap[0], self.pos[0])

  # i indice di un nodo
  # RETURN: indice del nodo padre
  # se il nodo in posizione i e' la radice, non fa nulla
  def parent(self, i):
      if i == 0:
          return None
      return (i + 1) // 2 - 1

  # i indice di un nodo
  # RETURN: indice del figlio sinistro
  def left(self, i):
      j = i * 2 + 1
      if j >= len(self.heap):
          return None
      return j

  # i indice di un nodo
  # REUTRN: indice del figlio destro
  def right(self, i):
      j = i * 2 + 2
      if j >= len(self.heap):
          return None
      return j    

  # estrae il nodo radice dalla heap
  def extract(self):
      self.heap[0] = self.heap[-1]
      self.pos[0] = self.pos[-1]
      self.heap = self.heap[:-1]
      self.pos = self.pos[:-1]
      self.heapify(0)

  # x nodo da inserire
  # inserisce il nodo nella heap
  def insert(self, x, p):
      self.heap.append(x)
      self.pos.append(p)
      self.moveup(len(self.heap) - 1)

  # i indice di un nodo della heap
  # x nodo da scambiare
  # scambia il nodo in posizione i nella heap con il nodo x
  def change(self, i, x):
      assert i >= 0 and i < len(self.heap)
      if x > self.heap[i]:
          self.heap[i] = x
          self.moveup(i)
      elif x < self.heap[i]:
          self.heap[i] = x
          self.heapify(i)

  # a array di interi
  # costruisce una minheap dall'array a 
  def buildheap(self, a):
      self.heap = a.copy()
      for i in range(len(self.heap) - 1, -1, -1):
          self.heapify(i)

  # i indice del nodo da cui parte la procedura
  # "sistema" l'albero in modo che il sottoalbero con randice il
  # nodo in posizione i sia una minheap
  def heapify(self, i):
      l = self.left(i)
      r = self.right(i)
      argmax = i
      if l != None and self.heap[l] > self.heap[argmax]:
          argmax = l
      if r != None and self.heap[r] > self.heap[argmax]:
          argmax = r
      if i != argmax:
          self.heap[i], self.heap[argmax] = self.heap[argmax], self.heap[i]
          self.pos[i], self.pos[argmax] = self.pos[argmax], self.pos[i]
          self.heapify(argmax)

  # i indice di un nodo
  # scambia il nodo in posizione i nella heap con il suo genitore
  # se il nodo in posizione i e' la radice, non fa nulla
  def moveup(self, i):
      if i == 0:
          return
      p = self.parent(i)
      if p != None and self.heap[i] > self.heap[p]:
          self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
          self.pos[i], self.pos[p] = self.pos[p], self.pos[i]
          self.moveup(p)


# ^ MEDIANOFMEDIANS ^ #

# a array di interi
# i indice dell'elemento da trovare
def median_of_medians_select(a, i):
  return select2(a, 0, len(a)-1, i)

# a array di interi
# p inizio dell'array
# r fine dell'array
# i indice dell'elemento da trovare
def select2(a,p,r,i):

  while (r-p+1) % 5 != 0:     # ripeto finche' non ho un numero di elementi multiplo di 5
    if r - p != 0:
      for j in range(p, r+1): # metto il minimo in prima posizione
        if a[p] > a[j]:
          a[p], a[j] = a[j], a[p]
    if i == 1:                # se i = 1 ho finito
      return a[p]
    p = p + 1                 # se i != 1 lavoro sul resto dell'array
    i = i - 1

  g = int((r - p + 1) / 5)              # numero dei gruppi da 5 elementi (e' un intero)
  if g == 1:
    array_bubble_sort(a, p, g)
  else:
    for j in range(p, p+g):       
      array_bubble_sort(a, j, g)        # sorting per ogni gruppo
                                        # bubblesort / quicksort (deve essere IN PLACE)
  x = select2(a, p+2*g, p+3*g, -(-g//2)) # trovo median of medians
  q = partition_around(a, p, r, x)

  k = q - p + 1                         # indice effettivo (senza contare lo zero)
  if i == k:
    return a[q]                         # il pivot e' il risultato
  elif i > k:
    return select2(a, q+1, r, i-k)
  else:
    return select2(a, p, q-1, i)

# a array di interi
# j indice del gruppo
# g numero dei gruppi
# esegue bubblesort sul gruppo j-esimo dell'array a
def array_bubble_sort(a, j, g):
   for i in range(0, 4):
      for k in range(0, 4):
         if a[j+k*g] > a[j+(k+1)*g]:
            a[j+k*g], a[j+(k+1)*g] = a[j+(k+1)*g], a[j+k*g]

# a array di interi
# p inizio dell'array
# r fine dell'array
# scambia x con l'ultimo elemento e richiama Partition "normale"
# sull'intervallo [p,r] dell'array a
def partition_around(a, p, r, x):
  i = p
  while a[i] != x:
    i += 1
  a[i], a[r] = a[r], a[i]
  return partition(a, p, r)

# a array di interi
# p inizio dell'array
# r fine dell'array
# esegue partition sull'intervallo [p,r] dell'array a
def partition(a, p, r):
  x = a[r]
  u = p - 1
  for v in range(p, r-1):
    if a[v] <= x:
      u = u + 1
      a[u], a[v] = a[v], a[u]
  a[u+1], a[r] = a[r], a[u+1] 
  return u + 1   # indice di median of medians alla fine di Partition


# ^ EXECUTION TIME ^ #

# genera array di dimensione n con interi compresi nell'intervallo [0, maxv] 
def genera_input(n, maxv):
    a = [0] * n
    for i in range(n):
        a[i] = i
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
def benchmark(n, maxv, k, func, risoluzione, max_rel_error=0.001):
    tmin = risoluzione * ( 1 + ( 1 / max_rel_error ) )
    count = 0
    start = time.perf_counter()
    while (time.perf_counter() - start) < tmin:
        a = genera_input(n, maxv)
        if k == 0: k = len(a)
        if len(a) > 0:  # Check if array is empty
            func(a, k)
        count += 1
    duration = time.perf_counter() - start
    return duration/count

# Test
if __name__=="__main__":
    risoluzione = resolution()
    nmin = 100
    nmax = 100000
    iters = 100     # quante volte genera un array di tale dim, migliora la precisione
    base = 2 ** ( (math.log(nmax) - math.log(nmin)) / (iters-1) )

    points = [(None, None, None, None, None, None)] * iters

    for i in range(iters):
        print(f"\r{i}",end='')
        n = int(nmin * (base ** i))
        points[i] = (n, benchmark(n, n, 1, randomized_quickselect, risoluzione, 0.001),
                        benchmark(n, n, 1, quickselect, risoluzione, 0.001),
                        benchmark(n, n, 0, min_heapselect, risoluzione, 0.001),
                        benchmark(n, n, 0, minmax_heapselect, risoluzione, 0.001),
                        benchmark(n, n, 0, median_of_medians_select, risoluzione, 0.001))

# Plot
xs, ys1, ys2, ys3, ys4, ys5= zip(*points)
nxs = np.array(xs)

# TEMPI DI ESECUZIONE NEL CASO PEGGIORE

# Randomized Quickselect
fig1, ax1 = plt.subplots()
fig1.suptitle("Tempo di esecuzione Randomized QuickSelect")
ax1.scatter(xs, ys1, color='#A5B592', s=10)
ax1.plot(xs, ys1, color='#A5B592', label='Randomized QuickSelect')
coeff1 = np.polyfit(xs, ys1, 2)
fit1 = np.poly1d(coeff1)
ax1.plot(xs, fit1(xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='Θ(n²)') # Th(n^2)
ax1.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax1.grid(True)
ax1.legend(loc='upper left')

fig11, ax11 = plt.subplots()
fig11.suptitle("Tempo di esecuzione Randomized QuickSelect (log)")
ax11.scatter(xs, ys1, color='#A5B592', s=10)
ax11.plot(xs, ys1, color='#A5B592', label='Randomized QuickSelect')
ax11.plot(xs, fit1(xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='Θ(n²)') # Th(n^2)
ax11.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax11.grid(True)
ax11.legend(loc='upper left')
ax11.set_xscale("log")
ax11.set_yscale("log")

# Quickselect
fig2, ax2 = plt.subplots()
fig2.suptitle("Tempo di esecuzione QuickSelect")
plt.scatter(xs, ys2, color='#A23E48', s=10)
plt.plot(xs, ys2, color='#A23E48', label='QuickSelect')
coeff2 = np.polyfit(xs, ys2, 2)
fit2 = np.poly1d(coeff2)
ax2.plot(xs, fit2(xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='Θ(n²)') # Th(n^2)
ax2.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax2.grid(True)
ax2.legend(loc='upper left')

fig12, ax12 = plt.subplots()
fig12.suptitle("Tempo di esecuzione QuickSelect (log)")
plt.scatter(xs, ys2, color='#A23E48', s=10)
plt.plot(xs, ys2, color='#A23E48', label='QuickSelect')
ax12.plot(xs, fit2(xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='Θ(n²)') # Th(n^2)
ax12.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax12.grid(True)
ax12.legend(loc='upper left')
ax12.set_xscale("log")
ax12.set_yscale("log")

# Heapselect
fig3, ax3 = plt.subplots()
fig3.suptitle("Tempo di esecuzione HeapSelect")
plt.scatter(xs, ys3, color='#4A6C6F', s=10)
plt.plot(xs, ys3, color='#4A6C6F', label='HeapSelect')
coeff3 = np.polyfit(xs+np.log(xs)*xs,ys3,1)
fit3 = np.poly1d(coeff3)
ax3.plot(xs, fit3(np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='O(n + k logk)') # O(n+klogk)
ax3.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax3.grid(True)
ax3.legend(loc='upper left')

fig13, ax13 = plt.subplots()
fig13.suptitle("Tempo di esecuzione Heapselect (log)")
plt.scatter(xs, ys3, color='#4A6C6F', s=10)
plt.plot(xs, ys3, color='#4A6C6F', label='HeapSelect')
ax13.plot(xs, fit3(np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='O(n + k logk)') # O(n+klogk)
ax13.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax13.grid(True)
ax13.legend(loc='upper left')
ax13.set_xscale("log")
ax13.set_yscale("log")

# Minmax_Heapselect
fig4, ax4 = plt.subplots()
fig4.suptitle("Tempo di esecuzione Minmax HeapSelect")
plt.scatter(xs, ys4, color='#3BB273', s=10)
plt.plot(xs, ys4, color='#3BB273', label='Minmax_HeapSelect')
coeff4 = np.polyfit(xs+np.log(xs)*xs,ys4,1)
fit4 = np.poly1d(coeff4)
ax4.plot(xs, fit4(xs+np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50 , label='O(n + k logk)') # O(n+klogk)
ax4.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax4.grid(True)
ax4.legend(loc='upper left')

fig14, ax14 = plt.subplots()
fig14.suptitle("Tempo di esecuzione Minmax Heapselect (log)")
plt.scatter(xs, ys4, color='#3BB273', s=10)
plt.plot(xs, ys4, color='#3BB273', label='Minmax_HeapSelect')
ax14.plot(xs, fit4(xs+np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50 , label='O(n + k logk)') # O(n+klogk)
ax14.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax14.grid(True)
ax14.legend(loc='upper left')
ax14.set_xscale("log")
ax14.set_yscale("log")

# Median of medians select 
fig5, ax5 = plt.subplots()
fig5.suptitle("Tempo di esecuzione MedianOfMediansSelect")
plt.scatter(xs, ys5, color='#FFAD0A', s=10)
plt.plot(xs, ys5, color='#FFAD0A', label='MedianOfMediansSelect')
a5, b5 = np.polyfit(xs, ys5, 1)
ax5.plot(xs, a5*nxs+b5, color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='O(n)') # O(n)
ax5.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax5.grid(True)
ax5.legend(loc='upper left')

fig15, ax15 = plt.subplots()
fig15.suptitle("Tempo di esecuzione MedianOfMediansSelect (log)")
plt.scatter(xs, ys5, color='#FFAD0A', s=10)
plt.plot(xs, ys5, color='#FFAD0A', label='MedianOfMediansSelect')
ax15.plot(xs, a5*nxs+b5, color='#404040', linestyle='dashed', linewidth=2.5, alpha=0.50, label='O(n)') # O(n)
ax15.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax15.grid(True)
ax15.legend(loc='upper left')
ax15.set_xscale("log")
ax15.set_yscale("log")

# GRAFICO COMPLETO
fig6, ax6 = plt.subplots()
fig6.suptitle("Grafico di comparazione")

plt.scatter(xs, ys2, color='#A5B592', label='Randomized QuickSelect', s=10)
ax6.plot(xs, ys2, color='#A5B592', linestyle='-', linewidth=2.5)

plt.scatter(xs, ys3, color='#4A6C6F', label='HeapSelect', s=10)
ax6.plot(xs, ys3, color='#4A6C6F', linestyle='-', linewidth=2.5)

plt.scatter(xs, ys5, color='#FFAD0A', label='MedianodMediansSelect', s=10)
ax6.plot(xs, ys5, color='#FFAD0A', linestyle='-', linewidth=2.5)

ax6.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax6.grid(True)
ax6.legend(loc='upper left')

fig16, ax16 = fig6, ax6
fig16.suptitle("Grafico di comparazione (log)")
plt.scatter(xs, ys2, color='#A5B592', label='Randomized QuickSelect', s=10)
ax16.plot(xs, ys2, color='#A5B592', linestyle='-', linewidth=2.5)

plt.scatter(xs, ys3, color='#4A6C6F', label='HeapSelect', s=10)
ax16.plot(xs, ys3, color='#4A6C6F', linestyle='-', linewidth=2.5)

plt.scatter(xs, ys5, color='#FFAD0A', label='MedianodMediansSelect', s=10)
ax16.plot(xs, ys5, color='#FFAD0A', linestyle='-', linewidth=2.5)

ax16.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax16.grid(True)
ax16.legend(loc='upper left')
ax16.set_xscale("log")
ax16.set_yscale("log")

# GRAFICO QUICKSELECT E RANDOMIZED QUICKSELECT
fig7, ax7 = plt.subplots()
fig7.suptitle("Comparazione tempo di esecuzione QuickSelect e Randomized QuickSelect")
plt.scatter(xs, ys2, color='#A5B592')
plt.scatter(xs, ys1, color='#A23E48')
ax7.plot(xs, ys2, color='#A5B592', linestyle='-', linewidth=2.5, label='QuickSelect')
ax7.plot(xs, ys1, color='#A23E48', linestyle='-', linewidth=2.5, label='Randomized QuickSelect')
ax7.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax7.grid(True)
ax7.legend(loc='upper left')

# GRAFICO HEAPSELECT E VARIANTE
fig8, ax8 = plt.subplots()
fig8.suptitle("Comparazione Heapselect con variante")
plt.scatter(xs, ys3, color='#4A6C6F', s=10)
plt.scatter(xs, ys4, color='#3BB273', s=10)
ax8.plot(xs, ys3, color='#4A6C6F', linestyle='-', linewidth=2.5, label='HeapSelect')
ax8.plot(xs, ys4, color='#3BB273', linestyle='-', linewidth=2.5,  label='Minmax HeapSelect')
ax8.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax8.grid(True)
ax8.legend(loc='upper left')

plt.show()
plt.close()
