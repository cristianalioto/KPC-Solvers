import os
# Impostiamo i thread a 1 per le librerie numeriche per evitare conflitti nel multiprocessing sui file.
# CP-SAT ignora questo e userà tutti i core disponibili quando lanciato sequenzialmente.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.getcwd())
import time
import json
import multiprocessing

from src.utilities.plot import generate_all_plots
from src.utilities.stats import generate_all_stats
from src.utilities.input_loader import parse_file
from src.solvers.cp_solver import KPC_CPSolver
from src.solvers.grasp_solver import KPC_GRASPSolver
from src.solvers.mip_solver import KPC_MIPSolver


# ==========================================
# Utilità Generali
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_dataset_selection():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"\n => Errore CRITICO: La cartella '{data_dir}' non esiste.")
        sys.exit(1)

    available_datasets = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')],
        key=lambda x: (x[0], int(x[1:])) if len(x) > 1 and x[1:].isdigit() else x
    )

    print(f"\nDataset disponibili: {', '.join(available_datasets)}")
    while True:
        d = input(f">> Seleziona Dataset: ").strip().upper()
        if d in available_datasets:
            return d
        print(f"\n => Input non valido.")

def print_table_header(solver_name, suffix=""):
    full_name = f"{solver_name} {suffix}".strip()
    header = f"{'PROG':<11} | {'FILE':<40} | {full_name + ' (TIME)':<15} | {full_name + ' (OBJ)':<15}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    return header

def print_table_row(i, total, filename, time_val, obj_val, status=None):
    obj_str = f"{int(obj_val)}"
    if status == 'OPTIMAL': obj_str += "*"
    print(f"{f'({i}/{total})':<11} | {filename:<40} | {time_val:<15.3f} | {obj_str:<15}")

###
# Utilità per GRASP & Warm Start
###
def check_solution_validity(data, selected_items):
    if not selected_items: return True, "Empty"
    current_weight = sum(data['weights'][i] for i in selected_items)
    if current_weight > data['capacity']: return False, "OverCapacity"
    sel_set = set(selected_items)
    for u, v in data['conflicts']:
        if u in sel_set and v in sel_set: return False, "Conflict"
    return True, "OK"

def load_precomputed_grasp(report_dir, dataset):
    """Carica i risultati GRASP se esistono per usarli come Warm Start"""
    grasp_file = os.path.join(report_dir, f"{dataset}_GRASP.json")
    lookup = {}
    if os.path.exists(grasp_file):
        try:
            with open(grasp_file, 'r') as f:
                data = json.load(f)
            for item in data.get("results", []):
                lookup[item['filename']] = item['grasp']
        except Exception:
            pass  # Silently fail if file corrupt or empty
    return lookup

def resolve_grasp_solution(data, filename, precomputed_grasp, ws_mode):
    """
    Restituisce la soluzione GRASP:
    1. Dal dizionario precalcolato se esiste.
    2. Calcolandola al volo se ws_mode=True ma non esiste il precalcolo.
    3. None se ws_mode=False.
    """
    if not ws_mode:
        return None, 0

    # Caso 1: C'è il precalcolato
    if precomputed_grasp and filename in precomputed_grasp:
        g_res = precomputed_grasp[filename]
        return g_res.get('selected_items', []), g_res.get('objective', 0)

    # Caso 2: Calcolo al volo (Fallback)
    solver = KPC_GRASPSolver(data, max_iterations=50)
    res = solver.solve()
    return res.get('selected_items', []), res.get('objective', 0)

###
# Workers (Multiprocessing)
###
def run_grasp_worker(args):
    filepath, filename = args
    data = parse_file(filepath)
    if not data: return None

    solver = KPC_GRASPSolver(data, max_iterations=100)
    res = solver.solve()

    valid, _ = check_solution_validity(data, res.get('selected_items', []))
    if not valid: res['objective'] = 0

    return {"filename": filename, "n": data['n'], "grasp": res}

def run_mip_worker(args):
    filepath, filename, ws_mode, precomputed_grasp_item = args
    data = parse_file(filepath)
    if not data: return None

    sol_list, sol_val = None, None
    if ws_mode:
        if precomputed_grasp_item:
            sol_list = precomputed_grasp_item.get('selected_items')
            sol_val = precomputed_grasp_item.get('objective')
        else:
            g_solver = KPC_GRASPSolver(data, max_iterations=50)
            g_res = g_solver.solve()
            sol_list = g_res.get('selected_items')
            sol_val = g_res.get('objective')

    solver = KPC_MIPSolver(data, time_limit_seconds=60)
    res = solver.solve(sol_list, sol_val)

    return {"filename": filename, "n": data['n'], "mip": res, "ws_used": ws_mode}

###
# Managers
###

def run_grasp_manager(dataset, data_path, files, max_workers, report_dir):
    out_file = os.path.join(report_dir, f"{dataset}_GRASP.json")
    print(f"\n--> AVVIO GRASP su {dataset} (Parallel: {max_workers} workers)...")
    print_table_header("GRASP")

    tasks = [(os.path.join(data_path, f), f) for f in files]
    results = []

    start_t = time.time()
    with multiprocessing.Pool(processes=max_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_grasp_worker, tasks), 1):
            if res:
                results.append(res)
                print_table_row(i, len(files), res['filename'], res['grasp']['time'], res['grasp']['objective'])
    total_time = time.time() - start_t

    with open(out_file, "w") as f:
        json.dump({"config": {"dataset": dataset, "mode": "GRASP", "total_time": total_time}, "results": results}, f,
                  indent=4)
    print(f" => Salvato: {out_file}")

def run_cp_manager(dataset, data_path, files, report_dir, ws_mode):
    mode_suffix = "WARM" if ws_mode else "COLD"
    out_file = os.path.join(report_dir, f"{dataset}_CP_{mode_suffix}.json")

    grasp_lookup = {}
    if ws_mode:
        grasp_lookup = load_precomputed_grasp(report_dir, dataset)
        if not grasp_lookup:
            print("   [!] Warning: File GRASP non trovato. Calcolo GRASP al volo attivo.")

    print(f"\n--> AVVIO CP [{mode_suffix}] su {dataset} (Sequenziale)...")
    print_table_header("CP", mode_suffix)

    results = []
    start_t = time.time()

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(data_path, filename)
        data = parse_file(filepath)
        if not data: continue

        sol_list, sol_val = resolve_grasp_solution(data, filename, grasp_lookup, ws_mode)

        # CP Solver (Usa tutti i thread internamente)
        solver = KPC_CPSolver(data, time_limit_seconds=60)
        res = solver.solve(sol_list, sol_val)

        results.append({"filename": filename, "n": data['n'], "cp": res, "ws_used": ws_mode})
        print_table_row(i, len(files), filename, res['time'], res['objective'], res.get('status'))

    total_time = time.time() - start_t
    with open(out_file, "w") as f:
        json.dump({"config": {"dataset": dataset, "mode": f"CP_{mode_suffix}", "ws": ws_mode, "total_time": total_time},
                   "results": results}, f, indent=4)
    print(f" => Salvato: {out_file}")

def run_mip_manager(dataset, data_path, files, max_workers, report_dir, ws_mode):
    mode_suffix = "WARM" if ws_mode else "COLD"
    out_file = os.path.join(report_dir, f"{dataset}_MIP_{mode_suffix}.json")

    grasp_lookup = {}
    if ws_mode:
        grasp_lookup = load_precomputed_grasp(report_dir, dataset)

    print(f"\n--> AVVIO MIP [{mode_suffix}] su {dataset} (Parallel: {max_workers} workers)...")
    print_table_header("MIP", mode_suffix)

    tasks = []
    for f in files:
        g_item = grasp_lookup.get(f) if ws_mode else None
        tasks.append((os.path.join(data_path, f), f, ws_mode, g_item))

    results = []
    start_t = time.time()

    with multiprocessing.Pool(processes=max_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_mip_worker, tasks), 1):
            if res:
                results.append(res)
                print_table_row(i, len(files), res['filename'], res['mip']['time'], res['mip']['objective'],
                                res['mip'].get('status'))

    total_time = time.time() - start_t
    with open(out_file, "w") as f:
        json.dump(
            {"config": {"dataset": dataset, "mode": f"MIP_{mode_suffix}", "ws": ws_mode, "total_time": total_time},
             "results": results}, f, indent=4)
    print(f" => Salvato: {out_file}")

def run_complete_benchmark(max_workers):
    target_datasets = ["C1", "C3", "C10", "R1", "R3", "R10"]
    print("\n=======================================================")
    print("      AVVIO AUTOMATICO COMPLETO (ALL DATASETS)       ")
    print("=======================================================")
    print(f"Datasets: {', '.join(target_datasets)}")
    print(f"Workers MIP/GRASP: {max_workers}")
    print(f"CP Sequenziale: Full workers per solver")
    print("-------------------------------------------------------")
    print("ATTENZIONE: Questa operazione richiederà molto tempo.")
    input("Premi INVIO per iniziare...")

    for dataset in target_datasets:
        data_path = os.path.join("data", dataset)
        if not os.path.exists(data_path):
            print(f"\n[SKIP] Dataset {dataset} non trovato in 'data/'.")
            continue

        files = sorted([f for f in os.listdir(data_path) if f.startswith("BPPC")])
        report_dir = os.path.join("outputs", "reports", dataset)
        os.makedirs(report_dir, exist_ok=True)

        print(f"\n\n#######################################################")
        print(f"             ELABORAZIONE DATASET: {dataset}")
        print(f"#######################################################")

        # 1. Esegui GRASP
        run_grasp_manager(dataset, data_path, files, max_workers, report_dir)

        # 2. Esegui CP (COLD & WARM)
        run_cp_manager(dataset, data_path, files, report_dir, ws_mode=False)  # Cold
        run_cp_manager(dataset, data_path, files, report_dir, ws_mode=True)  # Warm

        # 3. Esegui MIP (COLD & WARM)
        run_mip_manager(dataset, data_path, files, max_workers, report_dir, ws_mode=False)  # Cold
        run_mip_manager(dataset, data_path, files, max_workers, report_dir, ws_mode=True)  # Warm

    print("\n\n=======================================================")
    print("           TUTTE LE OPERAZIONI COMPLETATE.             ")
    print("=======================================================")

###
# Main
###
def main():
    clear_screen()
    print("==========================================")
    print("            KPC BENCHMARK TOOL            ")
    print("==========================================")
    print("  1. Esegui GRASP (Parallel)")
    print("  2. Esegui CP    (Sequential)")
    print("  3. Esegui MIP   (Parallel)")
    print("  4. ESEGUI TUTTO (C1, C3, C10, R1... - COLD & WARM)")
    print("  ----------------------------------------")
    print("  5. Genera Statistiche")
    print("  6. Genera Grafici")
    print("  0. Esci")
    print("==========================================")

    # Configurazione Workers
    max_workers = multiprocessing.cpu_count()

    try: choice = int(input(">> Scelta: "))
    except: choice = -1

    match choice:
        case 0: print("Uscita."); return
        case 4: run_complete_benchmark(max_workers); return
        case 5: generate_all_stats(); return
        case 6: generate_all_plots(); return

    # Opzioni Singole (1, 2, 3)
    dataset = get_dataset_selection()
    data_path = os.path.join("data", dataset)
    files = sorted([f for f in os.listdir(data_path) if f.startswith("BPPC")])
    report_dir = os.path.join("outputs", "reports", dataset)
    os.makedirs(report_dir, exist_ok=True)

    match choice:
        case 1: run_grasp_manager(dataset, data_path, files, max_workers, report_dir)
        case 2:
            ws = input(">> Warm Start con GRASP? (y/n): ").strip().lower() == 'y'
            run_cp_manager(dataset, data_path, files, report_dir, ws_mode=ws)
        case 3:
            ws = input(">> Warm Start con GRASP? (y/n): ").strip().lower() == 'y'
            run_mip_manager(dataset, data_path, files, max_workers, report_dir, ws_mode=ws)

if __name__ == "__main__":
    main()