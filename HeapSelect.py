# Algoritmo HeapSelect con complessita'
# Th(n) sia nel caso pessimo che nel caso medio

# a array
# p indice di inizio array
# r indice di fine array
# k indice dell'elemento da trovare

def heap_select(a, k):
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


# ^ IMPLEMENTAZIONE ^

def input_array():
  return [int(x) for x in input().split(" ") if x]

a = input_array()
k = int(input())
print( heap_select(a, k) )
