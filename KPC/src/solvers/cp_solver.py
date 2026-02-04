from ortools.sat.python import cp_model
import time


class KPC_CPSolver:
    def __init__(self, data, time_limit_seconds=60):
        self.n = data['n']
        self.profits = data['profits']
        self.weights = data['weights']
        self.capacity = data['capacity']
        self.conflicts = data['conflicts']
        self.time_limit = time_limit_seconds

    def solve(self, initial_solution=None, initial_solution_value=None):
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(self.n)]           # Variabili
        objective_expr = cp_model.LinearExpr.WeightedSum(x, self.profits) # Definizione obiettivo
        model.Maximize(objective_expr)                                    # Massimizzare il profitto

        ### Warm Start
        if initial_solution and initial_solution_value and initial_solution_value > 0:
            model.Add(objective_expr >= int(initial_solution_value)) # Lower Bound Injection (Cut)
            for i in initial_solution:
                model.AddHint(x[i], 1) # Hinting: non gli zeri, il solver gestisce bene la sparsità

        ### Vincoli
        model.Add(cp_model.LinearExpr.WeightedSum(x, self.weights) <= self.capacity) # Peso Tot <= Capacità
        for u, v in self.conflicts:
            model.AddBoolOr([x[u].Not(), x[v].Not()]) # Conflitti (clausole booleane)

        ### Configurazione Solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        # solver.parameters.num_search_workers = 8 # Di default li usa tutti
        #solver.parameters.log_search_progress = True

        ### Risoluzione
        start_time = time.time()
        status_val = solver.Solve(model)
        elapsed = time.time() - start_time
        status_name = solver.StatusName(status_val)

        ### Recupero Risultati
        selected_items = []
        objective_val : int = 0
        try:
            best_bound = int(solver.BestObjectiveBound())
        except:
            best_bound = 0

        # Se ha trovato almeno una soluzione (Ottima o Ammissibile)
        if status_val == cp_model.OPTIMAL or status_val == cp_model.FEASIBLE:
            objective_val = int(solver.ObjectiveValue())
            for i in range(self.n):
                if solver.Value(x[i]) == 1:
                    selected_items.append(i)

        return {
            "status": status_name,            # "OPTIMAL", "FEASIBLE", "UNKNOWN"
            "objective": objective_val,       # Il miglior valore trovato (LB) - Intero
            "upper_bound": best_bound,        # Il limite teorico (UB) - Intero
            "time": elapsed,
            "selected_items": selected_items  # Oggetti della soluzione migliore
        }