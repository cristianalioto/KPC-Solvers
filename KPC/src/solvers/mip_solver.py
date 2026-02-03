from ortools.linear_solver import pywraplp


class KPC_MIPSolver:
    def __init__(self, data, time_limit_seconds=60):
        self.n = data['n']
        self.profits = data['profits']
        self.weights = data['weights']
        self.capacity = data['capacity']
        self.conflicts = data['conflicts']
        self.time_limit = time_limit_seconds

    def solve(self, initial_solution=None, initial_solution_value=None):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return {'status': 'ERROR_NO_SCIP', 'objective': 0, 'upper_bound': 0, 'time': 0, 'selected_items': []}

        solver.SetTimeLimit(int(self.time_limit * 1000))
        x = [solver.IntVar(0, 1, f'x_{i}') for i in range(self.n)] # Variabili

        ### Vincoli
        solver.Add(sum(self.weights[i] * x[i] for i in range(self.n)) <= self.capacity) # Capacità
        for u, v in self.conflicts:
            solver.Add(x[u] + x[v] <= 1) # Conflitti

        ### Obiettivo
        objective_expr = sum(self.profits[i] * x[i] for i in range(self.n))
        solver.Maximize(objective_expr)

        ### Warm Start
        if initial_solution and initial_solution_value and initial_solution_value > 0:
            solver.Add(objective_expr >= initial_solution_value) # Lower Bound Injection (Cut)
            ## Hinting
            initial_set = set(initial_solution)
            hint_vars = x
            hint_vals = [1.0 if i in initial_set else 0.0 for i in range(self.n)]
            solver.SetHint(hint_vars, hint_vals) # SCIP richiede due liste: [variabili], [valori]

        ### Risoluzione e Risultati
        status = solver.Solve()
        status_name = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.NOT_SOLVED: "UNKNOWN"
        }.get(status, "UNKNOWN")

        objective_val: int = 0
        selected_items = []
        try:
            best_bound = int(solver.Objective().BestBound()) # Upper Bound (teorico)
        except:
            best_bound = 0

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            objective_val = int(solver.Objective().Value() + 1e-9) # +1e-9 per sicurezza sui float

            for i in range(self.n):
                # x[i].solution_value() restituisce float (0.0 o 1.0)
                if x[i].solution_value() > 0.5:
                    selected_items.append(i)

        return {
            "status": status_name,
            "objective": objective_val,
            "upper_bound": best_bound,
            "time": solver.wall_time() / 1000.0,
            "selected_items": selected_items
        }