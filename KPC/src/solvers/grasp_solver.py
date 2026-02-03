import random
import time


class KPC_GRASPSolver:
    def __init__(self, data, max_iterations=50, alpha=0.8):
        self.n = data['n']
        self.profits = data['profits']
        self.weights = data['weights']
        self.capacity = data['capacity']
        self.conflicts = data['conflicts']

        self.max_iterations = max_iterations
        self.alpha = alpha

        # Usiamo una lista di set per i conflitti => molto veloce per controllare le adiacenze
        self.adj = [set() for _ in range(self.n)]
        for u, v in self.conflicts:
            self.adj[u].add(v)
            self.adj[v].add(u)

        # Pre-calcoliamo i gradi per l'euristica
        self.degrees = [len(self.adj[i]) for i in range(self.n)]

    def _calculate_greedy_scores(self):
        """
        Calcola il punteggio di appetibilità per ogni oggetto.
        Più alto è il punteggio, più vogliamo prenderlo.
        """
        scores = []
        for i in range(self.n):
            w = self.weights[i] if self.weights[i] > 0 else 1e-5
            # Formula: (Profitto / Peso) ridotto se l'oggetto ha molti conflitti
            score = (self.profits[i] / w) / (1.0 + 0.5 * self.degrees[i])
            scores.append((score, i))

        # Ordiniamo dal migliore al peggiore
        scores.sort(key=lambda x: x[0], reverse=True) # Ordine "decrescente" di "score" ([0])
        return scores

    def _constructive_phase(self, sorted_candidates):
        """
        Costruisce una soluzione da zero.
        Usa 'forbidden' per ricordarsi chi non può più essere preso.
        """
        solution = set()
        current_weight = 0
        forbidden = set()  # Insieme dei nodi vietati (già presi o conflittuali)
        candidates_pool = sorted_candidates # Copiamo il riferimento alla lista ordinata

        ### Verrà
        while True:
            rcl = [] # Restricted Candidate List
            current_max_score = -1.0
            first_valid_idx = -1

            # Scansioniamo i candidati per trovare il max score valido
            # (grazie all'ordinamento, il primo valido che troviamo è il massimo)
            for idx, (score, item) in enumerate(candidates_pool):
                if (item in forbidden) or (current_weight + self.weights[item] > self.capacity):
                    continue

                # Troviamo il miglior score DISPONIBILE al momento
                current_max_score = score
                first_valid_idx = idx
                break

            # Se non abbiamo trovato nessuno valido, la costruzione è finita
            if current_max_score < 0: break

            # Definiamo la soglia per entrare nella RCL
            limit = current_max_score * self.alpha

            # Riempie la RCL partendo dal primo valido trovato
            # Si ferma appena lo score scende sotto il limite (perché la lista è ordinata!)
            for i in range(first_valid_idx, len(candidates_pool)):
                score, item = candidates_pool[i]
                if score < limit: break  # Inutile continuare, gli altri sono peggiori
                if item not in forbidden:
                    if current_weight + self.weights[item] <= self.capacity:
                        rcl.append(item)

            if not rcl: break

            selected = random.choice(rcl) # Scelta casuale

            # Aggiungiamo alla soluzione
            solution.add(selected)
            current_weight += self.weights[selected]

            # Aggiorniamo i divieti: vietiamo il nodo selezionato E tutti i suoi vicini
            forbidden.add(selected)
            forbidden.update(self.adj[selected])

        return list(solution), current_weight

    def _local_search(self, solution_list, current_weight):
        """
        Cerca di migliorare la soluzione facendo scambi (Swap) o aggiunte (Add).
        """
        solution = set(solution_list)
        current_profit = sum(self.profits[i] for i in solution)

        # Lista di oggetti fuori dalla soluzione (potenziali candidati)
        all_nodes = set(range(self.n))
        non_selected = list(all_nodes - solution)

        ### Itera fino a che non trova più soluzioni migliori
        improved = True
        while improved:
            improved = False
            random.shuffle(non_selected)  # Mischia per variare la ricerca

            ## 1. ADD (prova ad aggiungere)
            # Cerca oggetti che entrano "gratis" (senza togliere nulla)
            to_remove = []
            for j in non_selected:
                if current_weight + self.weights[j] <= self.capacity:
                    if self.adj[j].isdisjoint(solution): # Veloce nel controllare conflitti tra set
                        solution.add(j)
                        current_weight += self.weights[j]
                        current_profit += self.profits[j]
                        to_remove.append(j)
                        improved = True
                        # Continua a cercare altre aggiunte nello stesso ciclo (Greedy Packing)

            # Pulizia lista candidati
            for item in to_remove:
                non_selected.remove(item)

            if improved: continue  # Se ha aggiunto, ricomincia il ciclo

            ## 2. SWAP (prova a scambiare)
            # Proviamo a inserire 'j' (fuori) togliendo un 'i' (dentro) che lo blocca
            for j in non_selected:
                conflicts_in_sol = solution.intersection(self.adj[j]) # Trova chi blocca j dentro la soluzione

                # Se j è bloccato da ESATTAMENTE UN elemento 'i'
                if len(conflicts_in_sol) == 1:
                    i = list(conflicts_in_sol)[0]

                    # Controlla se lo scambio conviene => se il profitto aumenta
                    diff_profit = self.profits[j] - self.profits[i]
                    if diff_profit > 0:
                        diff_weight = self.weights[j] - self.weights[i]

                        # Controlla se lo scambio non viola i vincoli di capacità
                        if current_weight + diff_weight <= self.capacity:
                            # Eseguiamo lo scambio
                            solution.remove(i)
                            solution.add(j)
                            current_weight += diff_weight
                            current_profit += diff_profit
                            non_selected.remove(j)
                            non_selected.append(i)
                            improved = True
                            break  # Riparte dal ciclo principale

        ### Termine -> quando smette di migliorare
        return list(solution), current_profit

    def solve(self):
        start_time = time.time()
        best_solution = []
        best_profit = 0

        ### Calcolo punteggi euristici
        scores = self._calculate_greedy_scores()

        ### Iterazioni di GRASP
        for _ in range(self.max_iterations):
            # A. Costruzione (Greedy Randomized)
            cand_sol_list, cand_weight = self._constructive_phase(scores)

            # B. Ricerca Locale (Ottimizzazione)
            final_sol_list, final_profit = self._local_search(cand_sol_list, cand_weight)

            # C. Aggiornamento del Best Found
            if final_profit > best_profit:
                best_profit = final_profit
                best_solution = final_sol_list
                # print(f"New Best found: {best_profit}")

        ### Termine
        elapsed_time = time.time() - start_time
        return {
            "status": "FEASIBLE",
            "objective": best_profit,
            "time": elapsed_time,
            "selected_items": best_solution
        }