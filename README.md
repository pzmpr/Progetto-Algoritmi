# ^ ALGORITMI DA ANALIZZARE ^ #

 - Quick Select               |   complessità O(n2) nel caso pessimo e Θ(n) nel caso medio
                              |
 - Heap Select                |   complessità O(n+klogk) sia nel caso pessimo che nel caso medio
                              | 
 - Median of Medians Select   |   complessità Θ(n) sia nel caso pessimo che nel caso medio



# ^ PROGETTO DI LABORATORIO ^ #

1. tempi di esecuzione (test)
2. come si comportano i 3 algoritmi
3. benchmark: genero input a caso e misuro il tempo e esecuzione dei 3 algoritmi
4. metto a grafico i risultati con scala lineare e scala logaritmica (matplot)
5. identificare cose interessanti dal benchmark
    es. dipendenza da un algoritmo (quanto dipende quicksort da partition o dalle chiamate ricorsive, dipendenza da k...)



# ^ CLOCK PYTHON ^ # (OLD)

    time
      \_____ monotonic()  : clock di sistema monotono
	  
1. Stimare la RISOLUZIONE del clock di sistema
        utilizzando un ciclo while per calcolare l'intervallo minimo di tempo misurabile

    import time
    ...
    def resolution():
        start = time.monotonic()
        while time.monotonic() == start:
            pass
        stop = time.monotonic()
        return stop - start

2. In funzione della risoluzione stimata R e dell'errore relativo massimo ammissibile (E=0.001), si calcola il tempo minimo misurabile

    Tmin = R*(1/E + 1)

3. Per stimare il TEMPO MEDIO di ESECUZIONE di un algoritmo su istanze dell'input di dimensione n, si utilizza un ciclo while, iterando l'esecuzione dell'algoritmo su un input di dimensione n,
   generato in modo pseudo-casuale, e misurando complessivamente un intervallo di tempo superiore a Tmin. La misurazione deve essere effettuata senza interrompere il clock, ovvero
   calcolando l'intero intervallo di tempo trascorso dall'inizio dell'iterazione fino al momento il cui il tempo misurato risulti superiore a Tmin. 
   Il tempo medio di esecuzione per una singola istanza di input sarà quindi ottenuto calcolando il RAPPORTO fra il tempo totale misurato e il numero di iterazioni dell'algoritmo eseguite
   (questa divisione non influisce sull'errore relativo commesso)

Si consiglia l'uso di GRAFICI comparativi, sia in scale LINEARI - n vs t(n) - che doppiamente LOGARITMICHE - log(n) vs log(t(n))


Esempio
n = dim array -> funzione di i
n = a * b**i progressione geometrica, andamento esponenziale
for i from 0 to 99
    quando i = 0,  n = 100
    quando i = 99, n = 100000
nmin = 100, nmax = 100000
i = 0 -> nmin, i=99 -> 100000      100 = a * b**0,  100000 = a * b**100
a = nmin,   
nmin * b**99 = nmax,   log(nmin) + log(99) = log(nmax),   log(b) = log(nmax) - log(nmin)   ...     

A = nmin
B = 2 ** (( math.log(max) - math.log(nmin) / (iters - 1)))
n = int(A * (B ** i))

a = generatearray(n)
start = timenow() -> (*) il clock del sistema non cresce in modo monotono nel tempo perché sincronizzato con un server
algoritmo(a)         usare un clock monotono, adatto per cronometrare (monotonioc clock)
end = timenow()      Python time, monotonic clock
t = end - start      (*) Altro problema: i tempi di esecuzione sono brevi, la risoluzione di esecuzione dello strumento è piccola
                        cercare di limitare l'errore di esecuzione (rapporto tra valore assoluto e relativo (?))
                        
Come limitare l'errore relativo:
x = tempo di esecuzione da stimare
~x = tempo misurato
ð = risoluzione del clock (1mm, 1ms, ...), determinato dall'hardware
x - ~x <= ð
epsylon = errore relativo = ð / x
epsylon < 0.001

#! ð / x <= 0.001 = epsylonmax
epsylonmax = errore relativo massimo ammesso
delta <= x * epsylonmax
~x-ð <= x <= ~x+ð ?
~x >= ð,   1/epsylonmax 
 
 
 
# ^ RELAZIONE ^ #

no pseudocodice
no spiegazione dell'algoritmo
spiegare scelte particolari (es. partition random o no)

* SCALETTA *
- Tempi di esecuzione nei casi medi
- Comparazione Quickselect e Randomized Quickselect (average)
- Comparazione Heapselect e Minmax Heapselect (average)
- Tempi di esecuzione nei casi peggiori
- Comparazione Quickselect e Randomized Quickselect (worst)
- Comparazione Heapselect e Minmax Heapselect (worst)

* CASI PEGGIORI *
test nei casi peggiori eseguiti con array ad hoc ordinati in 
ordine crescente e k che varia in base al contesto
(per quickselect k=1, per il resto k=len(array))

Quickselect: - array ordinato crescente e k = 1
             - array ordinato decrescente e k = len(a)
			
Heapselect: - caso minheap -> a ordinato e k = len(a)
            - caso maxheap -> a ordinato e k = 1
			
MedianOfMedians: - a ordinato e k=len(a)
                 - se k = 1 -> caso favorevole se len(array) non
				               e' multiplo di 5

* QUICKSELECT NEL CASO PEGGIORE *
Because partitioning n elements takes Theta(n) time,
the recurrence for the worst-case running time is the same as for QUICKSORT:
T(n) = T(n-1) + Theta(n) with the solution T(n) = Theta(n^2) We’ll see that the algorithm
has a linear expected running time, however, and because it is randomized,
no particular input elicits the worst-case behavior.

* COMPARAZIONE QUICKSELECT E RANDOMIZED QUICKSELECT *
caso medio -> uguale
caso peggiore -> leggermente meglio randomized quickselect (per il partition randomizzato)
				   la complessita' rimane comunque Theta(n^2)

* COMPARAZIONE HEAPSELECT E MINMAX HEAPSELECT *
caso medio -> meglio minmax heapselect
caso peggiore -> - simili per array ordinato e k = 1
                 - molto meglio minmax per array ordinato e k = len(a) 
					         (si riconduce al caso semplice di maxheapselect)



 # ^ CASO PESSIMO DI MEDIAN OF MEDIANS ^

Funzione che genera un array per il caso "quasi pessimo" di median of medians select:
	l'array e' generato in modo che alla prima iterazione l'algoritmo venga richiamato
 	su 7/10 dell'array. Tuttavia questo vale solamente per la prima iterazione.

	def genera_input_mofm(n):
	    a = [0] * n
	    g = len(a) // 5
	    r = len(a) % 5
	    for _ in range(r):
	        a[r] = 0
	    for i in range(r,n):
	        if i % g >= -(-g//2):
	            a[i] = len(a) + i
	        else:
	            a[i] = i
	    return a 

L'algoritmo eseguito con questa funzione impiega tempi simili all'esecuzione con input randomico (caso medio) e tempi peggiori rispetto all'esecuzione con array ordinato

Teoricamente l'algoritmo nel caso pessimo verrebbe richiamato a ogni iterazione su 7/10
dell'array. In pratica questo però non accade perché il partition tende a "sistemare"
l'array in modo che la prossima chiamata ricada su un caso piu' favorevole
 
 --- RISPOSTA DI CHAT GPT ---

Analisi Ricorsiva

La selezione del pivot divide l'array in tre parti, una delle quali contiene al massimo il 
70% degli elementi originali, il che garantisce un progresso lineare.

Caso Peggiore Pratico

Il caso peggiore si verifica quando la riduzione dell'array non è uniforme e l'elemento 
cercato è sempre nella parte più grande della suddivisione. Tuttavia, anche in questo 
scenario, il comportamento dell'algoritmo median of medians garantisce che ogni passo 
riduca significativamente il problema.

Conclusione

Anche nei casi peggiori teorici, l'algoritmo median of medians select riduce progressivamente 
la dimensione del problema ad ogni passo, assicurando che il tempo totale rimanga lineare. 
Quindi, nonostante possiamo costruire un caso dove sembra che il partizionamento sia 
sfavorevole, l'algoritmo è progettato per mantenere O(n) tempo di esecuzione grazie alla 
scelta robusta del pivot.
