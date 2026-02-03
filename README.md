# 🎒 KPC Solver: Knapsack Problem with Conflicts

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OR-Tools](https://img.shields.io/badge/Library-Google%20OR--Tools-4285F4?logo=google&logoColor=white)](https://developers.google.com/optimization)
![MIP](https://img.shields.io/badge/Exact_Solver-MIP-FF6F00)
![CP-SAT](https://img.shields.io/badge/Exact_Solver-CP--SAT-FF6F00)
![GRASP](https://img.shields.io/badge/Metaheuristic-GRASP-F4B400)

[![University](https://img.shields.io/badge/University-UNIMORE-003366)](https://www.unimore.it/)
![Course](https://img.shields.io/badge/Course-Algoritmi_di_Ottimizzazione-009E73)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Progetto per il corso di Algoritmi di Ottimizzazione** <br>
> **Anno Accademico:** 2025/2026 <br>
> **Autore:** Cristian Piero Alioto <br>
> **Università:** Università degli Studi di Modena e Reggio Emilia (FIM)

-----

## Descrizione
Questo progetto affronta il **Knapsack Problem with Conflicts (KPC)**, una variante NP-Hard del classico problema dello zaino in cui esistono vincoli di incompatibilità tra coppie di oggetti, rappresentati da un grafo G=(V,E).

L'obiettivo è analizzare e confrontare le performance di **solutori esatti** e **metaeuristiche**, valutando l'efficacia di strategie ibride come il **Warm Start** su un ampio benchmark di **4.320 istanze**.

-----

## Metodologie Implementate
Il software integra tre approcci risolutivi principali e una tecnica di ibridazione:

### 1. Solutori Esatti (Google OR-Tools)
* **MIP (Mixed Integer Programming):** Modella i conflitti come disuguaglianze lineari. Efficace su istanze con vincoli di capacità stretti.
* **CP-SAT (Constraint Programming):** Utilizza tecniche SAT avanzate (*Lazy Clause Generation*). Modella i conflitti come clausole booleane, risultando superiore nelle istanze ad alta densità di conflitti.

### 2. Metaeuristica GRASP
Implementazione della **Greedy Randomized Adaptive Search Procedure** con due fasi:
* **Constructive Phase:** Costruzione probabilistica basata su uno score dinamico che penalizza gli oggetti con molti conflitti.
* **Local Search:** Miglioramento *First Improvement* tramite mosse **ADD** e **SWAP**.

### 3. Strategia Ibrida: Warm Start
Tecnica per accelerare i solutori esatti iniettando informazioni ottenute da GRASP:
* **Lower Bound Cut:** Impone al solver di cercare solo soluzioni migliori di quella euristica.
* **Solution Hinting:** Fornisce la soluzione generata da GRASP come punto di partenza per guidare il branching iniziale.

-----

## Struttura del Repository
```text
KPC-Solver/
├── data/               # Benchmark (4320 istanze)
├── outputs/            # Risultati Raw (JSON), Statistiche (JSON) e Grafici/Tabelle (PNG)
├── src/
│   ├── solvers/        # Implementazioni dei risolutori (MIP, CP, GRASP)
│   └── utilities/      # Input Loader, Generatori di statistiche e grafici
├── main.py             # Main
└── requirements.txt    # Dipendenze
```

-----

## Installazione e Utilizzo
1. **Clona il repository**

```bash
git clone https://github.com/cristianalioto/KPC-Solvers.git
cd KPC-Solver
```

2. **Installa le dipendenze**
Il progetto richiede `ortools`, `numpy` e `matplotlib`.

```bash
pip install -r requirements.txt
```

3. **Esegui il benchmark**
L'architettura usa il multiprocessing per parallelizzare l'esecuzione sui core disponibili.

```bash
python main.py
```

-----

## Licenza

Questo progetto è rilasciato sotto licenza MIT.

-----

Realizzato da Cristian Piero Alioto\
Data: 03/02/2026


