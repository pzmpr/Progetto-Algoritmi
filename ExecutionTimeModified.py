import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt


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

# a array
# p indice di inizio array
# r indice di fine array
# k indice dell'elemento da trovare
def heapselect(a, k):
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

  # i indice di un nofo
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

  # i indice di un nofo
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
        points[i] = (n, benchmark(n, 5*n, randomized_quickselect, risoluzione, 0.001),
                        benchmark(n, 5*n, quickselect, risoluzione, 0.001),
                        benchmark(n, 5*n, heapselect, risoluzione, 0.001),
                        benchmark(n, 5*n, median_of_medians_select, risoluzione, 0.001))

# Plot (line of best fit)
xs, ys1, ys2, ys3, ys4 = zip(*points)
nxs = np.array(xs)

fig1, ax1 = plt.subplots()
fig1.suptitle("Tempo di esecuzione Randomized QuickSelect")
plt.scatter(xs, ys1, color='#A5B592', s=10)
plt.plot(xs, ys1, color='#A5B592', label='andamento dell\'algoritmo')
a1, b1 = np.polyfit(xs, ys1, 1)
ax1.plot(xs, a1*nxs+b1, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n)') # O(n)
ax1.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax1.grid(True)
ax1.legend()

fig2, ax2 = plt.subplots()
fig2.suptitle("Tempo di esecuzione QuickSelect")
plt.scatter(xs, ys2, color='#A23E48', s=10)
plt.plot(xs, ys2, color='#A23E48', label='andamento dell\'algoritmo')
a2, b2 = np.polyfit(xs, ys2, 1)
ax2.plot(xs, a2*nxs+b2, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n)') # O(n)
ax2.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax2.grid(True)
ax2.legend()

fig3, ax3 = plt.subplots()
fig3.suptitle("Tempo di esecuzione HeapSelect")
plt.scatter(xs, ys3, color='#4A6C6F', s=10)
plt.plot(xs, ys3, color='#4A6C6F', label='andamento dell\'algoritmo')
coeff3 = np.polyfit(np.log(xs)*xs,ys4,1)
fit3 = np.poly1d(coeff3)
ax3.plot(xs, fit3(xs+np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, label='O(n + k logk)') # O(n+klogk)
ax3.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax3.grid(True)
ax3.legend( loc='upper left')

fig4, ax4 = plt.subplots()
fig4.suptitle("Tempo di esecuzione MedianOfMediansSelect")
plt.scatter(xs, ys4, color='#FFAD0A', s=10)
plt.plot(xs, ys4, color='#FFAD0A', label='andamento dell\'algoritmo')
a4, b4 = np.polyfit(xs, ys4, 1)
ax4.plot(xs, a4*nxs+b4, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n logn)') # O(nlogn)
ax4.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax4.grid(True)
ax4.legend()

comparason1n, ax5 = plt.subplots()
comparason1n.suptitle("Sovrapposizione dei tempi di esecuzione")
plt.plot(xs, ys1, color='#A5B592', label='Randomized QuickSelect')
plt.plot(xs, ys2, color='#A23E48', label='QuickSelect')
plt.plot(xs, ys3, color='#4A6C6F', label='HeapSelect')
plt.plot(xs, ys4, color='#FFAD0A', label='MedianOfMedians')
ax5.set(xlabel='Dimensione dell\'input (n)', ylabel='Tempo medio di esecuzione (secondi)')
ax5.grid(True)
ax5.legend()

#plt.show()
#plt.close()

# Scala logaritmica
fig1, ax1 = plt.subplots()
fig1.suptitle("Tempo di esecuzione Randomized QuickSelect")
plt.plot(xs, ys1, color='#A5B592', label='andamento dell\'algoritmo')
plt.scatter(xs, ys1, color='#A5B592', s=10)
a1, b1 = np.polyfit(xs, ys1, 1)
ax1.plot(xs, a1*nxs+b1, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n)') # O(n)
ax1.set(xlabel='Dimensione dell\'input (log)', ylabel='Tempo medio di esecuzione (secondi)')
plt.xscale('log')
plt.yscale('log')
ax1.grid(True)
ax1.legend()

fig2, ax2 = plt.subplots()
fig2.suptitle("Tempo di esecuzione QuickSelect")
plt.plot(xs, ys2, color='#A23E48', label='andamento dell\'algoritmo')
plt.scatter(xs, ys2, color='#A23E48', s=10)
a2, b2 = np.polyfit(xs, ys2, 1)
ax2.plot(xs, a2*nxs+b2, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n)') # O(n)
ax2.set(xlabel='Dimensione dell\'input (log)', ylabel='Tempo medio di esecuzione (secondi)')
plt.xscale('log')
plt.yscale('log')
ax2.grid(True)
ax2.legend()

fig3, ax3 = plt.subplots()
fig3.suptitle("Tempo di esecuzione HeapSelect")
plt.scatter(xs, ys3, color='#4A6C6F', s=10)
plt.plot(xs, ys3, color='#4A6C6F', label='andamento dell\'algoritmo')
coeff3 = np.polyfit(np.log(xs)*xs,ys4,1)
fit3 = np.poly1d(coeff3)
ax3.plot(xs, fit3(xs+np.log(xs)*xs), color='#404040', linestyle='dashed', linewidth=2.5, label='O(n + k logk)') # O(n+klogk)
ax3.set(xlabel='Dimensione dell\'input (log)', ylabel='Tempo medio di esecuzione (secondi)')
plt.xscale('log')
plt.yscale('log')
ax3.grid(True)
ax3.legend()

fig4, ax4 = plt.subplots()
fig4.suptitle("Tempo di esecuzione MedianOfMediansSelect")
plt.scatter(xs, ys4, color='#FFAD0A', s=10)
plt.plot(xs, ys4, color='#FFAD0A', label='andamento dell\'algoritmo')
a4, b4 = np.polyfit(xs, ys4, 1)
ax4.plot(xs, a4*nxs+b4, color='#404040', linestyle='dashed', linewidth=2.5, label='O(n logn)') # O(nlogn)
ax4.set(xlabel='Dimensione dell\'input (log)', ylabel='Tempo medio di esecuzione (secondi)')
plt.xscale('log')
plt.yscale('log')
ax4.grid(True)
ax4.legend()

comparason1log, ax5 = plt.subplots()
comparason1n.suptitle("Sovrapposizione dei tempi di esecuzione")
plt.plot(xs, ys1, color='#A5B592', label='Randomized QuickSelect')
plt.plot(xs, ys2, color='#A23E48', label='QuickSelect')
plt.plot(xs, ys3, color='#4A6C6F', label='HeapSelect')
plt.plot(xs, ys4, color='#FFAD0A', label='MedianOfMedians')
ax5.set(xlabel='Dimensione dell\'input (log)', ylabel='Tempo medio di esecuzione (secondi)')
plt.xscale('log')
plt.yscale('log')
ax5.grid(True)
ax5.legend()

plt.show()
plt.close()