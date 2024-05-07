# Algoritmo Median-of-Medians con complessita'
# O(n+klogk) sia nel caso pessimo che nel caso medio

# a array di interi
# i indice dell'elemento da trovare
def median_of_medians_select(a, i):
  return Select(a, 0, len(a)-1, i)

# a array di interi
# p inizio dell'array
# r fine dell'array
# i indice dell'elemento da trovare
def Select(a,p,r,i):
  
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
  x = Select(a, p+2*g, p+3*g, -(-g//2)) # trovo median of medians
  q = partition_around(a, p, r, x)

  k = q - p + 1                         # indice effettivo (senza contare lo zero)
  if i == k:
    return a[q]                         # il pivot e' il risultato
  elif i > k:
    return Select(a, q+1, r, i-k)
  else:
    return Select(a, p, q-1, i)

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


# ^ IMPLEMENTAZIONE ^

def input_array():
   return [int(x) for x in input().split(" ") if x]

a = input_array()
k = int(input())
print( median_of_medians_select(a, k) )
