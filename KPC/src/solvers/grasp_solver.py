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
        Prova a migliorare la soluzione cercando ottimi locali tramite due possibli mosse:
            1. ADD: Cerca di inserire oggetti se c'è spazio e nessun conflitto.
            2. SWAP: Prova a inserire un oggetto esterno rimuovendo TUTTI
               quelli interni che creano conflitto, purché il profitto migliori.
        """
        solution = set(solution_list)
        current_profit = sum(self.profits[i] for i in solution)

        # Set di tutti gli indici per calcolo veloce dei complementari
        all_nodes = set(range(self.n))

        improved = True
        while improved:
            improved = False

            # Identifichiamo i candidati (oggetti fuori dallo zaino)
            candidates = list(all_nodes - solution)
            random.shuffle(candidates)  # Mischia per variare l'esplorazione (tipico GRASP)

            ### FASE 1: ADD (Riempimento Greedy)
            # Proviamo prima ad aggiungere oggetti che non richiedono rimozioni
            items_added = False
            for j in candidates:
                # Controllo capacità
                if current_weight + self.weights[j] <= self.capacity:
                    # Controllo conflitti: disjoint = intersezione vuota con la soluzione
                    if self.adj[j].isdisjoint(solution):
                        solution.add(j)
                        current_weight += self.weights[j]
                        current_profit += self.profits[j]
                        improved = True
                        items_added = True
                        # Nota: non facciamo break qui, proviamo ad aggiungerne altri 'gratis'
                        # nello stesso passaggio per saturare lo zaino velocemente.

            if items_added:
                continue  # Se abbiamo aggiunto qualcosa, ricominciamo il ciclo principale

            ### FASE 2: SWAP (Logica "1-in, K-out")
            # Se siamo qui, non possiamo aggiungere nulla "gratis". Proviamo a forzare scambi.
            # Ricalcoliamo i candidati (quelli rimasti fuori dopo la fase ADD)
            candidates = list(all_nodes - solution)
            random.shuffle(candidates)

            for j in candidates:
                w_j = self.weights[j]
                p_j = self.profits[j]

                # Identifica TUTTI i conflitti che 'j' ha con gli oggetti nello zaino
                # self.adj[j] sono i vicini di j. L'intersezione ci dà chi dobbiamo buttare fuori.
                conflicts_in_bag = solution.intersection(self.adj[j])

                # Se non ci sono conflitti ma non siamo entrati nella fase ADD,
                # significa che non ci stava per peso. Non possiamo fare swap senza conflitti.
                if not conflicts_in_bag:
                    continue

                # Calcoliamo peso e profitto totale degli oggetti da rimuovere
                w_out = sum(self.weights[k] for k in conflicts_in_bag)
                p_out = sum(self.profits[k] for k in conflicts_in_bag)
                profit_gain = p_j - p_out # Guadagno netto di profitto?
                if profit_gain > 0:
                    new_weight = current_weight - w_out + w_j # Rispetto della capacità dopo lo scambio?
                    if new_weight <= self.capacity:
                        # APPLICA LA SWAP
                        solution.add(j)
                        solution -= conflicts_in_bag  # Rimuove tutti i conflittuali in un colpo solo
                        current_weight = new_weight
                        current_profit += profit_gain
                        improved = True
                        break  # First Improvement: appena troviamo uno scambio buono, riavviamo il ciclo

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