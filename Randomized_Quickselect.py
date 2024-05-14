import random

# Algoritmo QuickSelect con complessita'
# O(n) nel caso medio
# Th(n^2) nel caso pessimo
# Utilizza Randomized Partition che invece di utilizzare
# l'ultimo elemeno dell'array come perno ne sceglie uno 
# casualmente nell'intervallo dell'array

# a: array di interi
# p: indice di inizio array
# r: indice di fine array
# i: indice dell'elemento da trovare
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

# a:    array di interi
# low:  indice di inizio array
# high: indice di fine array
# sceglie un numero casuale nell'intervallo [low, high]
# scambia il numero con l'ultimo elemento di a
# infine richiama partition "normale"
def randomized_partition(a, low, high):
  i = random.randint(low, high)
  a[high], a[i] = a[i], a[high]
  return partition(a, low, high)

# a:    array di interi
# low:  inizio dell'array
# high: fine dell'array
# esegue partition sull'intervallo [low, high] dell'array a
def partition(a, low, high):
  p = a[high]
  i = low - 1
  for j in range(low, high):
    if a[j] <= p:
      i = i + 1
      a[i], a[j] = a[j], a[i]
  a[i+1], a[high] = a[high], a[i+1] 
  return i + 1


# ^ IMPLEMENTAZIONE ^

def input_array():
  return [int(x) for x in input().split(" ") if x]

a = input_array()
k = int(input())

print(randomized_quickselect(a, k))
