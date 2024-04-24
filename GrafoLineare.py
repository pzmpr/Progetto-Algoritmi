import time
import random
import matplotlib.pyplot as plt
import math

def partition(a, low, high):
  p = a[high - 1]
  i = low
  for j in range(low, high-1):
      if a[j] <= p:
          a[i], a[j] = a[j], a[i]
          i += 1
  a[i], a[high-1] = a[high-1], a[i]
  return i

def quicksort(a, low, high):
  if high - low <= 1:
      return
  middle = partition(a, low, high)
  quicksort(a, low, middle)
  quicksort(a, middle+1, high)

#def quicksort_select(a, k): # originale
  #assert k <= len(a)
  #quicksort(a, 0, len(a))
  #return a[k-1]

def quicksort_select(a, k): #fa piu veloce
    assert k <= len(a)
    sorted_array = sorted(a)
    return sorted_array[k-1]

#--------------------------------------------------------------------

# Funzione per generare input casuale
def genera_random_array(n):
    return [random.randint(1, 100000) for i in range(n)]

# Funzione per misurare il tempo di esecuzione
def resolution():
  start = time.monotonic()
  while time.monotonic() == start:
      pass
  stop = time.monotonic()
  return stop - start


# Funzione per misurare il tempo di esecuzione di quicksort_select
def misura_tempo_di_esecuzione(input_array, k):
    start_time = time.monotonic()
    quicksort_select(input_array, k)
    end_time = time.monotonic()
    return end_time - start_time

# Dimensioni dell'input da testare
grandezza_array = []
def trova_grandezza_array():
    A = 100
    B = math.pow(100000 / A, 1 / 99)
    array_lengths = []

    for i in range(100):
        n_i = int(A * math.pow(B, i))
        array_lengths.append(n_i)

    return array_lengths

grandezza_array= trova_grandezza_array()

# Numero di iterazioni per ogni dimensione dell'input
precisione = 50

# Risoluzione del clock di sistema
risoluzione = resolution()

# Test
dimensioni_input = []
tempi_medi = []

for dimensione in grandezza_array:
    tempo_totale = 0
    for _ in range(precisione):
        input_array = genera_random_array(dimensione)
        k = random.randint(1, dimensione)
        tempo_totale += misura_tempo_di_esecuzione(input_array, k)
    tempo_medio = tempo_totale / precisione
    dimensioni_input.append(dimensione)
    tempi_medi.append(tempo_medio)
    print(f"Dimensione dell'input: {dimensione}, Tempo medio di esecuzione: {tempo_medio:.6f} secondi")

plt.plot( dimensioni_input,tempi_medi, marker='o', linestyle='-')
plt.ylabel('Tempo medio di esecuzione (secondi)')
plt.xlabel('Dimensione dell\'input (n)')
plt.title('Quicksort Select - Tempi medi di esecuzione vs Dimensione dell\'input')
plt.grid(True)
plt.show()
