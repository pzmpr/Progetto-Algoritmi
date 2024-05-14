# Algoritmo QuickSelect con complessita'
# O(n) nel caso medio
# Th(n^2) nel caso pessimo

# a: array  di interi
# p: indice di inizio array
# r: indice di fine array
# i: indice dell'elemento da trovare
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

print(quickselect(a, k))


