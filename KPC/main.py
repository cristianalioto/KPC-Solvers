import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.getcwd()) # Aggiunge la directory corrente al path

import time
import json
import multiprocessing
import winsound

from src.utilities.plot import generate_all_plots
from src.utilities.stats import generate_stats, generate_all_stats
from src.utilities.input_loader import parse_file
from src.solvers.cp_solver import KPC_CPSolver
from src.solvers.grasp_solver import KPC_GRASPSolver
from src.solvers.mip_solver import KPC_MIPSolver


# ==========================================
# Utilità
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_dataset_selection():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"\n => Errore CRITICO: La cartella '{data_dir}' non esiste.")
        sys.exit(1)

    available_datasets = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])

    if not available_datasets:
        print(f"\n => Errore: Nessuna sottocartella trovata in '{data_dir}'.")
        sys.exit(1)

    print(f"\nDataset disponibili: {', '.join(available_datasets)}")
    while True:
        d = input(f">> Seleziona Dataset: ").strip().upper()
        if d in available_datasets:
            return d
        print(f"\n => Input non valido.")

def check_solution_validity(data, selected_items):
    """Controllo di validità di una soluzione"""
    if not selected_items: return True, "Empty"
    current_weight = sum(data['weights'][i] for i in selected_items)
    if current_weight > data['capacity']: return False, "OverCapacity"
    sel_set = set(selected_items)
    for u, v in data['conflicts']:
        if u in sel_set and v in sel_set: return False, "Conflict"
    return True, "OK"

# ==========================================
# Workers
# ==========================================
def run_grasp_worker(args):
    filepath, filename = args
    try:
        parts = filename.split('_')
        type_id = int(parts[1])
        density = float(parts[-1])
        data = parse_file(filepath)
    except:
        return None
    if not data: return None

    # Esegui GRASP
    solver = KPC_GRASPSolver(data, max_iterations=100)
    res = solver.solve()

    valid, _ = check_solution_validity(data, res.get('selected_items', []))
    if not valid: res['objective'] = 0

    return {
        "filename": filename,
        "n": data['n'],
        "type_id": type_id,
        "density": density,
        "grasp": res
    }

def run_full_benchmark_worker(args):
    filepath, filename, ws_mode, instance_class, precomputed_grasp = args
    try:
        parts = filename.split('_')
        type_id = int(parts[1])
        density = float(parts[-1])
        data = parse_file(filepath)
    except:
        return None
    if not data: return None

    ### Gestione GRASP: Se precalcolato usalo, altrimenti eseguilo
    if precomputed_grasp:
        res_grasp = precomputed_grasp
    else:
        solver = KPC_GRASPSolver(data, max_iterations=100)
        res_grasp = solver.solve()
        valid, _ = check_solution_validity(data, res_grasp.get('selected_items', []))
        if not valid: res_grasp['objective'] = 0

    ### Warm Start
    sol_list = res_grasp.get('selected_items', []) if ws_mode else None
    sol_val = res_grasp.get('objective', 0) if ws_mode else None

    ### Esecuzione
    TIME_LIMIT = 60
    res_mip = KPC_MIPSolver(data, time_limit_seconds=TIME_LIMIT).solve(sol_list, sol_val)
    res_cp = KPC_CPSolver(data, time_limit_seconds=TIME_LIMIT).solve(sol_list, sol_val)

    ### Risultati
    best_exact = max(res_cp.get('objective', 0), res_mip.get('objective', 0))
    gap = 100 * (best_exact - res_grasp['objective']) / best_exact if best_exact > 0 else 0
    return {
        "filename": filename, "n": data['n'], "type_id": type_id, "density": density,
        "ws_used": ws_mode, "gap": gap,
        "mip": res_mip, "cp": res_cp, "grasp": res_grasp
    }

# ==========================================
# Managers
# ==========================================
def run_grasp_manager(dataset, data_path, files, max_workers, report_dir):
    ### Configurazione
    out_file = os.path.join(report_dir, f"{dataset}_GRASP.json")
    print(f"\n--> AVVIO GRASP su {dataset}...")
    print(f" - Output: {out_file}")
    print(f" - Workers: {max_workers}")

    ### Header
    header = f"{'PROG':<11} | {'FILE':<45} | {'OBJ':<10} | {'TIME (s)':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    ### Multiprocessing
    tasks = [(os.path.join(data_path, f), f) for f in files]
    results = []
    start_t = time.time()
    with multiprocessing.Pool(processes=max_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_grasp_worker, tasks), 1):
            if res:
                results.append(res)
                filename = res['filename']
                obj_val = int(res['grasp']['objective'])
                time_val = res['grasp']['time']
                print(f"({i}/{len(files)})".ljust(11) +
                      f" | {filename:<45} | {obj_val:<10} | {time_val:<7.3f}")

    ### Termine esecuzione => Salvataggio
    total_time = time.time() - start_t
    with open(out_file, "w") as f:
        json.dump({
            "config": {"dataset": dataset, "mode": "GRASP", "total_time": total_time},
            "results": results
        }, f, indent=4)

    print("-" * len(header))
    print(f"\n => GRASP completato. File salvato in: {out_file}")

def run_full_benchmark_manager(dataset, data_path, files, max_workers, report_dir):
    instance_class = dataset[0]

    ### Check esistenza file GRASP precalcolato
    grasp_file_path = os.path.join(report_dir, f"{dataset}_GRASP.json")
    grasp_lookup = {}
    if os.path.exists(grasp_file_path):
        print(f"\n => Trovato file GRASP preesistente: {grasp_file_path}")
        print("    I risultati verranno riutilizzati per il calcolo del GAP e il Warm Start.")
        try:
            with open(grasp_file_path, 'r') as f:
                gdata = json.load(f)
            for item in gdata.get("results", []):
                grasp_lookup[item['filename']] = item['grasp']
        except Exception as e:
            print(f"\n => Errore lettura file GRASP: {e}. Verrà ricalcolato.")
            grasp_lookup = {}
    else:
        print("\n => Nessun file GRASP trovato. Verrà calcolato al volo per ogni istanza.")

    ### Warm Start?
    while True:
        ws_str = input("\n>> Warm Start? (y = SI / n = NO): ").strip().lower()
        if ws_str in ['y', 'n']: break
    ws_mode = (ws_str == 'y')

    ### Configurazioni
    file_suffix = "WARM" if ws_mode else "COLD"
    out_file = os.path.join(report_dir, f"{dataset}_{file_suffix}.json")

    ### Header
    print(f"\n--> AVVIO FULL BENCHMARK [{file_suffix}] su {dataset}...")
    print(f" - Output: {out_file}")
    print(f" - Workers: {max_workers}")
    header = (f"{'PROG':<11} | {'FILE':<40} | {'WS':<3} | "
              f"{'MIP':<9} | {'T(MIP)':<7} | {'CP':<9} | {'T(CP)':<7} | {'GRASP':<9} | {'GAP%':<6}")

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    ### Multiprocessing
    tasks = []
    for f in files:
        pre_g = grasp_lookup.get(f)
        tasks.append((os.path.join(data_path, f), f, ws_mode, instance_class, pre_g))
    results = []
    start_t = time.time()
    with multiprocessing.Pool(processes=max_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_full_benchmark_worker, tasks), 1):
            if res:
                results.append(res)
                rm, rc, rg = res['mip'], res['cp'], res['grasp']
                mip_s = f"{int(rm['objective'])}*" if rm.get('status') == 'OPTIMAL' else f"{int(rm['objective'])}"
                cp_s = f"{int(rc['objective'])}*" if rc.get('status') == 'OPTIMAL' else f"{int(rc['objective'])}"
                ws_str = "SI" if res['ws_used'] else "NO"
                prog_str = f"({i}/{len(files)})"
                print(f"{prog_str:<11} | {res['filename']:<40} | {ws_str:<3} | "
                      f"{mip_s:<9} | {rm['time']:<7.3f} | {cp_s:<9} | {rc['time']:<7.3f} | "
                      f"{int(rg['objective']):<9} | {res['gap']:<6.2f}")

    ### Termine Esecuzione => Salvataggio
    total_time = time.time() - start_t
    with open(out_file, "w") as f:
        json.dump({
            "config": {"dataset": dataset, "mode": file_suffix, "total_time": total_time},
            "results": results
        }, f, indent=4)

    print("-" * len(header))
    print(f"\n => Benchmark completato. File: {out_file}")
    generate_stats(dataset, file_suffix, results, total_time)  # Generazione automatica statistiche

# ==========================================
# Main
# ==========================================
def main():
    clear_screen()
    print("==========================================")
    print("            KPC BENCHMARK TOOL            ")
    print("==========================================")
    print("  1. Esegui GRASP")
    print("  2. Esegui FULL BENCHMARK (CP vs MIP vs GRASP)")
    print("  3. Genera Statistiche")
    print("  4. Genera Grafici")
    print("  0. Esci")
    print("==========================================")

    try:    choice = int(input(">> Scelta: "))
    except: choice = -1

    match choice:
        case 0: print("Uscita."); return
        case 3: generate_all_stats(); return
        case 4: generate_all_plots(); return

    # Selezione Dataset comune per opzione 1 e 2
    dataset = get_dataset_selection()
    data_path = os.path.join("data", dataset)
    files = sorted([f for f in os.listdir(data_path) if f.startswith("BPPC")])
    max_workers = int(multiprocessing.cpu_count() / 2 - 1)

    # Cartella output
    report_dir = os.path.join("outputs", "reports", dataset)
    os.makedirs(report_dir, exist_ok=True)

    match choice:
        case 1: run_grasp_manager(dataset, data_path, files, max_workers, report_dir)
        case 2: run_full_benchmark_manager(dataset, data_path, files, max_workers, report_dir)

    try:
        winsound.Beep(1000, 1000)
    except:
        pass

if __name__ == "__main__":
    main()