import time
import random
import matplotlib.pyplot as plt

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

def partition(a, low, high):
  p = a[high]
  i = low - 1
  for j in range(low, high):
    if a[j] <= p:
      i = i + 1
      a[i], a[j] = a[j], a[i]
  a[i+1], a[high] = a[high], a[i+1] 
  return i + 1
  
def randomized_partition(a, low, high):
  i = random.randint(low, high-1)
  a[high-1], a[i] = a[i], a[high-1]
  return partition(a, low, high)

#--------------------------------------------------------------------

# Funzione per generare input casuale
def genera_random_array(n):
    return [random.randint(1, 10000) for i in range(n)]

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
grandezza_array = [100, 1000, 100000]

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

plt.plot(tempi_medi, dimensioni_input, marker='o', linestyle='-')
plt.xlabel('Tempo medio di esecuzione (secondi)')
plt.ylabel('Dimensione dell\'input (n)')
plt.title('Quicksort Select - Tempi medi di esecuzione vs Dimensione dell\'input')
plt.grid(True)
plt.show()
