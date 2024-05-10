import random

# Algoritmo QuickSelect con complessita'
# O(n) nel caso medio
# Th(n^2) nel caso pessimo

# ^ VERSIONE STANDARD ^

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

# ^ VERSIONE CON RANDOMIZED_PARTITION ^

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


# ^ IMPLEMENTAZIONE ^

def input_array():
  return [int(x) for x in input().split(" ") if x]

a = input_array()
k = int(input())

# print(quickselect(a, k))
print(randomized_quickselect(a, k))

