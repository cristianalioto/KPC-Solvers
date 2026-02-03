import os
import json
from collections import defaultdict


def calculate_aggregated_stats(results_list):
    """Calcola le statistiche per un singolo file (COLD o WARM)."""
    stats_by_type = defaultdict(lambda: {
        'mip_time': [], 'mip_status': [],
        'cp_time': [], 'cp_status': [],
        'grasp_time': [], 'gap': []
    })

    stats_by_density = defaultdict(lambda: {
        'mip_time': [], 'mip_status': [],
        'cp_time': [], 'cp_status': [],
        'grasp_time': [], 'gap': []
    })

    for res in results_list:
        if 'type_id' not in res or 'density' not in res: continue

        # Estrazione dati
        m_stat = res.get('mip', {}).get('status', 'N/A')
        c_stat = res.get('cp', {}).get('status', 'N/A')
        m_time = res.get('mip', {}).get('time', 0)
        c_time = res.get('cp', {}).get('time', 0)
        g_time = res['grasp']['time']
        gap = res.get('gap', 0)

        # Aggregazione
        for key, container in [(res['type_id'], stats_by_type), (res['density'], stats_by_density)]:
            container[key]['mip_time'].append(m_time)
            container[key]['mip_status'].append(m_stat)
            container[key]['cp_time'].append(c_time)
            container[key]['cp_status'].append(c_stat)
            container[key]['grasp_time'].append(g_time)
            container[key]['gap'].append(gap)

    def compute_averages(source_dict, key_name):
        output_list = []
        for key, vals in sorted(source_dict.items()):
            count = len(vals['grasp_time'])
            if count == 0: continue

            output_list.append({
                key_name: key,
                "count": count,
                "avg_mip_time": sum(vals['mip_time']) / count,
                "mip_optimal_count": vals['mip_status'].count('OPTIMAL'),
                "avg_cp_time": sum(vals['cp_time']) / count,
                "cp_optimal_count": vals['cp_status'].count('OPTIMAL'),
                "avg_grasp_time": sum(vals['grasp_time']) / count,
                "avg_gap_pct": sum(vals['gap']) / count,
                "grasp_optimal_count": sum(1 for g in vals['gap'] if g <= 1e-9)  # Gap 0
            })
        return output_list

    return {
        "stats_by_type": compute_averages(stats_by_type, "type_id"),
        "stats_by_density": compute_averages(stats_by_density, "density")
    }


def calculate_comparison_stats(cold_results, warm_results):
    """Confronta WARM vs COLD e genera metriche differenziali."""

    # Mappa per filename per allineare le istanze
    cold_map = {r['filename']: r for r in cold_results}
    warm_map = {r['filename']: r for r in warm_results}

    common_files = set(cold_map.keys()) & set(warm_map.keys())

    # Strutture dati per aggregazione
    comp_by_type = defaultdict(lambda: defaultdict(list))
    comp_by_density = defaultdict(lambda: defaultdict(list))

    for fname in common_files:
        c = cold_map[fname]
        w = warm_map[fname]

        type_id = c['type_id']
        density = c['density']

        # Estrazione Valori
        mip_t_c, mip_t_w = c['mip']['time'], w['mip']['time']
        cp_t_c, cp_t_w = c['cp']['time'], w['cp']['time']

        mip_obj_c, mip_obj_w = c['mip']['objective'], w['mip']['objective']
        cp_obj_c, cp_obj_w = c['cp']['objective'], w['cp']['objective']

        grasp_gap = c.get('gap', 0)  # Grasp è uguale per entrambi, prendiamo da cold

        # Dati da raccogliere per ogni gruppo
        data_point = {
            'mip_time_cold': mip_t_c,
            'mip_time_warm': mip_t_w,
            'cp_time_cold': cp_t_c,
            'cp_time_warm': cp_t_w,
            'mip_obj_improved': 1 if mip_obj_w > mip_obj_c else 0,  # Warm ha trovato sol migliore
            'cp_obj_improved': 1 if cp_obj_w > cp_obj_c else 0,
            'mip_opt_cold': 1 if c['mip']['status'] == 'OPTIMAL' else 0,
            'mip_opt_warm': 1 if w['mip']['status'] == 'OPTIMAL' else 0,
            'cp_opt_cold': 1 if c['cp']['status'] == 'OPTIMAL' else 0,
            'cp_opt_warm': 1 if w['cp']['status'] == 'OPTIMAL' else 0,
            'grasp_optimal': 1 if grasp_gap <= 1e-9 else 0
        }

        # Aggiunta agli aggregatori
        for key, container in [(type_id, comp_by_type), (density, comp_by_density)]:
            for metric, value in data_point.items():
                container[key][metric].append(value)

    def compute_comp_averages(source_dict, key_name):
        output_list = []
        for key, vals in sorted(source_dict.items()):
            count = len(vals['mip_time_cold'])
            if count == 0: continue

            # Calcolo Medie Tempi
            avg_mip_c = sum(vals['mip_time_cold']) / count
            avg_mip_w = sum(vals['mip_time_warm']) / count
            avg_cp_c = sum(vals['cp_time_cold']) / count
            avg_cp_w = sum(vals['cp_time_warm']) / count

            # Calcolo Miglioramento % Tempo (Positivo = Warm è più veloce)
            # Gestione divisione per zero se avg_c è 0 (molto raro)
            mip_time_improv_pct = ((avg_mip_c - avg_mip_w) / avg_mip_c * 100) if avg_mip_c > 0 else 0
            cp_time_improv_pct = ((avg_cp_c - avg_cp_w) / avg_cp_c * 100) if avg_cp_c > 0 else 0

            # --- MODIFICA: Calcolo % di istanze migliorate nel profitto ---
            mip_better_count = sum(vals['mip_obj_improved'])
            cp_better_count = sum(vals['cp_obj_improved'])

            mip_obj_improv_pct = (mip_better_count / count) * 100
            cp_obj_improv_pct = (cp_better_count / count) * 100
            # -------------------------------------------------------------

            output_list.append({
                key_name: key,
                "count": count,

                # Tempi
                "avg_mip_time_cold": avg_mip_c,
                "avg_mip_time_warm": avg_mip_w,
                "mip_time_improvement_pct": mip_time_improv_pct,

                "avg_cp_time_cold": avg_cp_c,
                "avg_cp_time_warm": avg_cp_w,
                "cp_time_improvement_pct": cp_time_improv_pct,

                # Ottimalità (Conteggi assoluti)
                "mip_optimal_cold_count": sum(vals['mip_opt_cold']),
                "mip_optimal_warm_count": sum(vals['mip_opt_warm']),
                "cp_optimal_cold_count": sum(vals['cp_opt_cold']),
                "cp_optimal_warm_count": sum(vals['cp_opt_warm']),

                # Qualità Soluzione (Miglioramento Obiettivo)
                "mip_warm_better_obj_count": mip_better_count,
                "mip_obj_improvement_pct": mip_obj_improv_pct,  # <--- NUOVO CAMPO

                "cp_warm_better_obj_count": cp_better_count,
                "cp_obj_improvement_pct": cp_obj_improv_pct,  # <--- NUOVO CAMPO

                # GRASP
                "grasp_optimal_found_count": sum(vals['grasp_optimal'])
            })
        return output_list

    return {
        "comparison_by_type": compute_comp_averages(comp_by_type, "type_id"),
        "comparison_by_density": compute_comp_averages(comp_by_density, "density")
    }


def generate_stats(dataset, tag, results, total_time):
    """Genera stats singole e, se possibile, quelle comparative."""
    stats_dir = os.path.join("outputs", "stats", dataset)
    os.makedirs(stats_dir, exist_ok=True)
    report_dir = os.path.join("outputs", "reports", dataset)

    # 1. Genera Statistiche Singole (COLD o WARM)
    stats_path = os.path.join(stats_dir, f"{dataset}_{tag}_STATS.json")
    stats_data = calculate_aggregated_stats(results)

    with open(stats_path, "w") as f:
        json.dump({
            "config": {"dataset": dataset, "tag": tag, "total_time": total_time},
            "statistics": stats_data
        }, f, indent=4)
    print(f"    -> Statistiche singole generate: {dataset}_{tag}_STATS.json")

    # 2. Tenta di generare Statistiche Comparative (COMPARISON)
    path_cold = os.path.join(report_dir, f"{dataset}_COLD.json")
    path_warm = os.path.join(report_dir, f"{dataset}_WARM.json")

    if os.path.exists(path_cold) and os.path.exists(path_warm):
        # Evita di rigenerare il confronto due volte (se chiamato sia per Cold che per Warm)
        # Lo generiamo solo quando stiamo processando il WARM o se richiesto esplicitamente
        if tag == "WARM" or tag == "FINAL":
            print(f"    -> Trovati dati COLD e WARM: Generazione confronto...")
            try:
                with open(path_cold, 'r') as f:
                    res_cold = json.load(f)['results']
                with open(path_warm, 'r') as f:
                    res_warm = json.load(f)['results']

                comp_data = calculate_comparison_stats(res_cold, res_warm)

                comp_path = os.path.join(stats_dir, f"{dataset}_COMPARISON.json")
                with open(comp_path, "w") as f:
                    json.dump({
                        "config": {"dataset": dataset, "type": "WARM_VS_COLD_COMPARISON"},
                        "comparison": comp_data
                    }, f, indent=4)
                print(f"    => COMPARISON generato in: {comp_path}")
            except Exception as e:
                print(f"    *** Errore generazione confronto: {e}")


def generate_all_stats():
    """Scansiona tutte le cartelle in outputs/reports e genera le statistiche per tutti."""
    base_reports_dir = os.path.join("outputs", "reports")

    if not os.path.exists(base_reports_dir):
        print(" => Nessuna cartella 'outputs/reports' trovata.")
        return

    # Trova tutte le sottocartelle (C1, C2, R1, ecc...)
    datasets = sorted([d for d in os.listdir(base_reports_dir) if os.path.isdir(os.path.join(base_reports_dir, d))])

    if not datasets:
        print(" => Nessun dataset trovato.")
        return

    print(f"\nGenerazione statistiche massiva per: {', '.join(datasets)}")

    count = 1
    for dataset in datasets:
        print(f"\n ({count}/{len(datasets)}) Elaborazione Dataset: {dataset}...")
        dataset_path = os.path.join(base_reports_dir, dataset)

        # Filtra i file JSON validi (esclude GRASP puro, Stats già fatte e Comparison)
        files = [
            f for f in os.listdir(dataset_path)
            if f.endswith(".json") and "GRASP" not in f
        ]

        if not files:
            print(f"\n => Nessun file COLD/WARM trovato in {dataset}.")
            continue

        for fname in files:
            try:
                with open(os.path.join(dataset_path, fname), 'r') as f:
                    data = json.load(f)

                # Ricava il tag (es. "COLD" o "WARM") dal nome file (C1_COLD.json -> COLD)
                tag = fname.replace(f"{dataset}_", "").replace(".json", "")

                generate_stats(dataset, tag, data.get("results", []), data.get("config", {}).get("total_time", 0))
            except Exception as e:
                print(f"\n => Errore processando {fname}: {e}")
        count += 1