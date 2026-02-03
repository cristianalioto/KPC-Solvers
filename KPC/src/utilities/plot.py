import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


### Utilità
def load_json_data(filepath):
    """Carica i risultati dal JSON."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f).get('results', [])
    except Exception as e:
        print(f"Errore lettura {filepath}: {e}")
        return None

def aggregate_data(results, key_field):
    """Aggrega i tempi medi per MIP, CP e GRASP."""
    grouped = defaultdict(lambda: {'mip': [], 'cp': [], 'grasp': []})
    for row in results:
        k = row.get(key_field)
        if k is None: continue
        grouped[k]['mip'].append(row['mip'].get('time', 0))
        grouped[k]['cp'].append(row['cp'].get('time', 0))
        grouped[k]['grasp'].append(row['grasp'].get('time', 0))

    averaged = {}
    for k, v in grouped.items():
        averaged[k] = {
            'mip': sum(v['mip']) / len(v['mip']) if v['mip'] else 0,
            'cp': sum(v['cp']) / len(v['cp']) if v['cp'] else 0,
            'grasp': sum(v['grasp']) / len(v['grasp']) if v['grasp'] else 0
        }
    return averaged

### Grafici comparativi per dataset
def plot_solvers_comparison(data_cold, data_warm, dataset_name, out_dir):
    """Genera i grafici comparativi MIP vs CP vs GRASP per COLD e WARM."""
    modes = []
    if data_cold: modes.append(('COLD', data_cold))
    if data_warm: modes.append(('WARM', data_warm))

    for mode_name, data_source in modes:
        for group_key in ['type_id', 'density']:
            agg_data = aggregate_data(data_source, group_key)
            sorted_keys = sorted(agg_data.keys())
            if not sorted_keys: continue

            mip_vals = [agg_data[k]['mip'] for k in sorted_keys]
            cp_vals = [agg_data[k]['cp'] for k in sorted_keys]
            grasp_vals = [agg_data[k]['grasp'] for k in sorted_keys]

            x = np.arange(len(sorted_keys))
            width = 0.25

            plt.figure(figsize=(10, 6))
            plt.bar(x - width, mip_vals, width, label='MIP', color='#d62728')
            plt.bar(x, cp_vals, width, label='CP', color='#1f77b4')
            plt.bar(x + width, grasp_vals, width, label='GRASP', color='#2ca02c')

            xlabel_text = group_key.replace("_", " ").title()
            if group_key == 'density': xlabel_text = "Densità Conflitti"

            plt.xlabel(xlabel_text)
            plt.ylabel('Tempo Medio (s)')
            plt.title(f'{dataset_name} [{mode_name}] - Tempi per {xlabel_text}')
            plt.xticks(x, sorted_keys)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if max(mip_vals + cp_vals) > 10:
                plt.yscale('log')
                plt.ylabel('Tempo Medio (s) - Log Scale')

            filename = f"Compare_{mode_name}_{group_key}.png"
            plt.savefig(os.path.join(out_dir, filename))
            plt.close()

def plot_warm_vs_cold_impact(data_cold, data_warm, dataset_name, out_dir):
    """Genera confronto diretto COLD vs WARM per vedere il miglioramento."""
    if not data_cold or not data_warm: return

    for group_key in ['density', 'type_id']:
        agg_cold = aggregate_data(data_cold, group_key)
        agg_warm = aggregate_data(data_warm, group_key)

        keys = sorted(list(set(agg_cold.keys()) & set(agg_warm.keys())))
        if not keys: continue

        solvers = [('MIP', '#d62728'), ('CP', '#1f77b4')]

        for solver_name, color in solvers:
            key_lower = solver_name.lower()
            cold_vals = [agg_cold[k][key_lower] for k in keys]
            warm_vals = [agg_warm[k][key_lower] for k in keys]

            x = np.arange(len(keys))
            width = 0.35

            plt.figure(figsize=(10, 6))
            plt.bar(x - width / 2, cold_vals, width, label='Cold Start', color='gray', alpha=0.6)
            plt.bar(x + width / 2, warm_vals, width, label='Warm Start', color=color)

            label_map = {'density': 'Densità Conflitti', 'type_id': 'Tipo Istanza'}
            xlabel_text = label_map.get(group_key, group_key)

            plt.xlabel(xlabel_text)
            plt.ylabel('Tempo Medio (s)')
            plt.title(f'{dataset_name} - Impatto Warm Start su {solver_name} (per {xlabel_text})')
            plt.xticks(x, keys)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            filename = f"Impact_WarmStart_{solver_name}_{group_key}.png"
            plt.savefig(os.path.join(out_dir, filename))
            plt.close()

### Salvataggio Tabelle
def save_table_image_generic(data, col_labels, title, filename, output_dir):
    if not data: return

    row_height = 0.3
    fig_width = 10.0
    if len(col_labels) < 6: fig_width = 8.0

    fig_height = (len(data) + 1) * row_height + 0.2
    #fig_height = (len(data) + 1) * row_height + 0.8 # Con Titolo

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.95]
    )

    #plt.title(title, fontsize=12, fontweight='bold', pad=2, y=0.98)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_height(row_height * 1.2)
        elif row == len(data):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ffeb99')
            cell.set_height(row_height)
        else:
            if row % 2 == 0: cell.set_facecolor('#f2f2f2')
            cell.set_height(row_height)

    plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Generata tabella PNG: {filename}")

###  Genera sommari "globali" (WS Impact)
def prepare_data_with_average(raw_data, include_dataset_col=True):
    """Prepara i dati calcolando la media totale."""
    if not raw_data: return []

    # Calcolo Medie Totali (Ultima riga)
    # Per le tabelle aggregate (senza dataset), questa media è la 'media delle medie' dei gruppi
    # Per le tabelle dettagliate (con dataset), è la media di tutte le righe
    total_mip_time = sum(row['mip_time'] for row in raw_data)
    total_cp_time = sum(row['cp_time'] for row in raw_data)
    total_mip_obj = sum(row['mip_obj'] for row in raw_data)
    total_cp_obj = sum(row['cp_obj'] for row in raw_data)
    count = len(raw_data)

    avg_row_vals = {
        'mip_time': total_mip_time / count,
        'cp_time': total_cp_time / count,
        'mip_obj': total_mip_obj / count,
        'cp_obj': total_cp_obj / count
    }

    formatted_data = []
    last_dataset = None

    for row in raw_data:
        current_row = []
        if include_dataset_col:
            ds_label = row['dataset'] if row['dataset'] != last_dataset else ""
            current_row.append(ds_label)
            last_dataset = row['dataset']

        current_row.extend([
            row['key'],
            f"{row['mip_time']:.1f}%",
            f"{row['cp_time']:.1f}%",
            f"{row['mip_obj']:.1f}%",
            f"{row['cp_obj']:.1f}%"
        ])
        formatted_data.append(current_row)

    final_row = []
    if include_dataset_col: final_row.append("MEDIA")
    final_row.extend([
        "TOTALE",
        f"{avg_row_vals['mip_time']:.1f}%",
        f"{avg_row_vals['cp_time']:.1f}%",
        f"{avg_row_vals['mip_obj']:.1f}%",
        f"{avg_row_vals['cp_obj']:.1f}%"
    ])
    formatted_data.append(final_row)
    return formatted_data

def generate_global_summary_tables():
    """Genera tabelle riassuntive Comparative (Impact)."""
    base_stats_dir = os.path.join("outputs", "stats")
    if not os.path.exists(base_stats_dir): return

    datasets = sorted([d for d in os.listdir(base_stats_dir) if os.path.isdir(os.path.join(base_stats_dir, d))])

    raw_type_data_ds = []
    raw_density_data_ds = []

    global_type_acc = defaultdict(lambda: defaultdict(list))
    global_density_acc = defaultdict(lambda: defaultdict(list))

    print(f"\nGenerazione Tabelle Comparative (Impact)...")

    for dataset in datasets:
        comp_file = os.path.join(base_stats_dir, dataset, f"{dataset}_COMPARISON.json")
        if not os.path.exists(comp_file): continue
        try:
            with open(comp_file, 'r') as f:
                data = json.load(f)

            for row in data['comparison']['comparison_by_type']:
                # Accumulo per tabella dataset
                raw_type_data_ds.append({
                    'dataset': dataset, 'key': row['type_id'],
                    'mip_time': row['mip_time_improvement_pct'], 'cp_time': row['cp_time_improvement_pct'],
                    'mip_obj': row.get('mip_obj_improvement_pct', 0.0), 'cp_obj': row.get('cp_obj_improvement_pct', 0.0)
                })
                # Accumulo per tabella globale (per tipo)
                k = row['type_id']
                global_type_acc[k]['mip_time'].append(row['mip_time_improvement_pct'])
                global_type_acc[k]['cp_time'].append(row['cp_time_improvement_pct'])
                global_type_acc[k]['mip_obj'].append(row.get('mip_obj_improvement_pct', 0.0))
                global_type_acc[k]['cp_obj'].append(row.get('cp_obj_improvement_pct', 0.0))

            for row in data['comparison']['comparison_by_density']:
                raw_density_data_ds.append({
                    'dataset': dataset, 'key': row['density'],
                    'mip_time': row['mip_time_improvement_pct'], 'cp_time': row['cp_time_improvement_pct'],
                    'mip_obj': row.get('mip_obj_improvement_pct', 0.0), 'cp_obj': row.get('cp_obj_improvement_pct', 0.0)
                })
                k = row['density']
                global_density_acc[k]['mip_time'].append(row['mip_time_improvement_pct'])
                global_density_acc[k]['cp_time'].append(row['cp_time_improvement_pct'])
                global_density_acc[k]['mip_obj'].append(row.get('mip_obj_improvement_pct', 0.0))
                global_density_acc[k]['cp_obj'].append(row.get('cp_obj_improvement_pct', 0.0))

        except Exception as e:
            print(f"Errore lettura {comp_file}: {e}")

    def process_global_acc(accumulator):
        processed = []
        for key in sorted(accumulator.keys()):
            vals = accumulator[key]
            processed.append({
                'dataset': '', 'key': key,
                'mip_time': sum(vals['mip_time']) / len(vals['mip_time']),
                'cp_time': sum(vals['cp_time']) / len(vals['cp_time']),
                'mip_obj': sum(vals['mip_obj']) / len(vals['mip_obj']),
                'cp_obj': sum(vals['cp_obj']) / len(vals['cp_obj'])
            })
        return processed

    # Uso forced_avg=None, così ricalcolano la media sui dati che gli passiamo
    final_type_ds = prepare_data_with_average(raw_type_data_ds, include_dataset_col=True)
    final_density_ds = prepare_data_with_average(raw_density_data_ds, include_dataset_col=True)
    final_type_global = prepare_data_with_average(process_global_acc(global_type_acc), include_dataset_col=False)
    final_density_global = prepare_data_with_average(process_global_acc(global_density_acc), include_dataset_col=False)

    global_plots_dir = os.path.join("outputs", "plots", "SUMMARY_TABLES")
    os.makedirs(global_plots_dir, exist_ok=True)

    save_table_image_generic(final_type_ds, ["Dataset", "Type", "MIP Time %", "CP Time %", "MIP Obj %", "CP Obj %"],
                             "Impatto Warm Start (Type)", "Comparison_Impact_Type_By_Dataset.png", global_plots_dir)
    save_table_image_generic(final_density_ds,
                             ["Dataset", "Density", "MIP Time %", "CP Time %", "MIP Obj %", "CP Obj %"],
                             "Impatto Warm Start (Density)", "Comparison_Impact_Density_By_Dataset.png",
                             global_plots_dir)
    save_table_image_generic(final_type_global, ["Type", "MIP Time %", "CP Time %", "MIP Obj %", "CP Obj %"],
                             "Impatto Warm Start Medio (Type)", "Comparison_Impact_Type_Global.png", global_plots_dir)
    save_table_image_generic(final_density_global, ["Density", "MIP Time %", "CP Time %", "MIP Obj %", "CP Obj %"],
                             "Impatto Warm Start Medio (Density)", "Comparison_Impact_Density_Global.png",
                             global_plots_dir)

###  Genera sommari "singoli" (tempi e ottimi)
def extract_descriptive_stats(results, dataset_name):
    """Estrae statistiche descrittive (Tempi e Ottimi)."""
    stats_type = defaultdict(
        lambda: {'mip_time': [], 'cp_time': [], 'grasp_time': [], 'mip_opt': 0, 'cp_opt': 0, 'grasp_opt': 0,
                 'count': 0})
    stats_density = defaultdict(
        lambda: {'mip_time': [], 'cp_time': [], 'grasp_time': [], 'mip_opt': 0, 'cp_opt': 0, 'grasp_opt': 0,
                 'count': 0})

    for res in results:
        if 'type_id' not in res or 'density' not in res: continue
        m_time = res['mip'].get('time', 0)
        c_time = res['cp'].get('time', 0)
        g_time = res['grasp'].get('time', 0)
        m_opt = 1 if res['mip'].get('status') == 'OPTIMAL' else 0
        c_opt = 1 if res['cp'].get('status') == 'OPTIMAL' else 0
        g_opt = 1 if (res.get('gap', 100) < 1e-6) and (c_opt or m_opt) else 0

        for key, container in [(res['type_id'], stats_type), (res['density'], stats_density)]:
            container[key]['mip_time'].append(m_time)
            container[key]['cp_time'].append(c_time)
            container[key]['grasp_time'].append(g_time)
            container[key]['mip_opt'] += m_opt
            container[key]['cp_opt'] += c_opt
            container[key]['grasp_opt'] += g_opt
            container[key]['count'] += 1

    rows_type = []
    rows_density = []
    for container, out_list in [(stats_type, rows_type), (stats_density, rows_density)]:
        for key in sorted(container.keys()):
            data = container[key]
            n = data['count']
            if n == 0: continue
            out_list.append({
                'dataset': dataset_name, 'key': key,
                'mip_time': sum(data['mip_time']) / n, 'cp_time': sum(data['cp_time']) / n,
                'grasp_time': sum(data['grasp_time']) / n,
                'mip_opt': data['mip_opt'], 'cp_opt': data['cp_opt'], 'grasp_opt': data['grasp_opt'], 'total_inst': n
            })
    return rows_type, rows_density

def prepare_descriptive_table_data(raw_data):
    """Formatta i dati descrittivi per la tabella finale."""
    if not raw_data: return []

    n_rows = len(raw_data)
    # Media dei tempi, Somma degli ottimali
    avg_vals = {
        'mip_time': sum(r['mip_time'] for r in raw_data) / n_rows,
        'cp_time': sum(r['cp_time'] for r in raw_data) / n_rows,
        'grasp_time': sum(r['grasp_time'] for r in raw_data) / n_rows,
        'mip_opt': sum(r['mip_opt'] for r in raw_data),
        'cp_opt': sum(r['cp_opt'] for r in raw_data),
        'grasp_opt': sum(r['grasp_opt'] for r in raw_data),
        'total_inst': sum(r['total_inst'] for r in raw_data)
    }

    formatted = []
    last_ds = None
    for row in raw_data:
        ds_label = row['dataset'] if row['dataset'] != last_ds else ""
        formatted.append([
            ds_label, str(row['key']),
            f"{row['mip_time']:.2f}s", f"{row['cp_time']:.2f}s", f"{row['grasp_time']:.3f}s",
            f"{row['mip_opt']}/{row['total_inst']}", f"{row['cp_opt']}/{row['total_inst']}",
            f"{row['grasp_opt']}/{row['total_inst']}"
        ])
        last_ds = row['dataset']

    formatted.append([
        "TOTALE", "MEDIA",
        f"{avg_vals['mip_time']:.2f}s", f"{avg_vals['cp_time']:.2f}s", f"{avg_vals['grasp_time']:.3f}s",
        f"{avg_vals['mip_opt']}/{avg_vals['total_inst']}", f"{avg_vals['cp_opt']}/{avg_vals['total_inst']}",
        f"{avg_vals['grasp_opt']}/{avg_vals['total_inst']}"
    ])
    return formatted

def generate_descriptive_summary_tables():
    """Genera le tabelle descrittive."""
    base_reports_dir = os.path.join("outputs", "reports")
    if not os.path.exists(base_reports_dir): return

    datasets = sorted([d for d in os.listdir(base_reports_dir) if os.path.isdir(os.path.join(base_reports_dir, d))])

    cold_type_all, cold_dens_all = [], []
    warm_type_all, warm_dens_all = [], []

    print(f"\nGenerazione Tabelle Descrittive...")

    for dataset in datasets:
        cold_file = os.path.join(base_reports_dir, dataset, f"{dataset}_COLD.json")
        warm_file = os.path.join(base_reports_dir, dataset, f"{dataset}_WARM.json")

        if os.path.exists(cold_file):
            try:
                with open(cold_file, 'r') as f:
                    data = json.load(f).get('results', [])
                rows_t, rows_d = extract_descriptive_stats(data, dataset)
                cold_type_all.extend(rows_t)
                cold_dens_all.extend(rows_d)
            except:
                pass

        if os.path.exists(warm_file):
            try:
                with open(warm_file, 'r') as f:
                    data = json.load(f).get('results', [])
                rows_t, rows_d = extract_descriptive_stats(data, dataset)
                warm_type_all.extend(rows_t);
                warm_dens_all.extend(rows_d)
            except:
                pass

    data_cold_type = prepare_descriptive_table_data(cold_type_all)
    data_cold_dens = prepare_descriptive_table_data(cold_dens_all)
    data_warm_type = prepare_descriptive_table_data(warm_type_all)
    data_warm_dens = prepare_descriptive_table_data(warm_dens_all)

    global_plots_dir = os.path.join("outputs", "plots", "SUMMARY_TABLES")
    os.makedirs(global_plots_dir, exist_ok=True)

    cols_type = ["Dataset", "Type", "MIP T", "CP T", "GRASP T", "MIP Opt", "CP Opt", "GRASP Opt"]
    cols_dens = ["Dataset", "Density", "MIP T", "CP T", "GRASP T", "MIP Opt", "CP Opt", "GRASP Opt"]

    save_table_image_generic(data_cold_type, cols_type, "Statistiche Descrittive COLD START (per Tipo)",
                             "Descriptive_COLD_Type.png", global_plots_dir)
    save_table_image_generic(data_cold_dens, cols_dens, "Statistiche Descrittive COLD START (per Densità)",
                             "Descriptive_COLD_Density.png", global_plots_dir)
    save_table_image_generic(data_warm_type, cols_type, "Statistiche Descrittive WARM START (per Tipo)",
                             "Descriptive_WARM_Type.png", global_plots_dir)
    save_table_image_generic(data_warm_dens, cols_dens, "Statistiche Descrittive WARM START (per Densità)",
                             "Descriptive_WARM_Density.png", global_plots_dir)

### Genera ogni plot
def generate_all_plots():
    """Genera tutti i grafici per tutti i dataset presenti."""
    base_dir = os.path.join("outputs", "reports")
    if not os.path.exists(base_dir):
        print("Nessuna cartella report trovata.")
        return

    datasets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not datasets:
        print("Nessun dataset trovato.")
        return

    print(f"\nGenerazione Grafici per: {', '.join(datasets)}")

    count = 1
    for dataset in datasets:
        report_dir = os.path.join("outputs", "reports", dataset)
        plots_dir = os.path.join("outputs", "plots", dataset)
        os.makedirs(plots_dir, exist_ok=True)

        file_cold = os.path.join(report_dir, f"{dataset}_COLD.json")
        file_warm = os.path.join(report_dir, f"{dataset}_WARM.json")

        data_cold = load_json_data(file_cold)
        data_warm = load_json_data(file_warm)

        if not data_cold and not data_warm: continue
        print(f"{count}/{len(datasets)}: Grafici locali: {dataset}")
        plot_solvers_comparison(data_cold, data_warm, dataset, plots_dir)
        if data_cold and data_warm:
            plot_warm_vs_cold_impact(data_cold, data_warm, dataset, plots_dir)
        count += 1

    generate_global_summary_tables()
    generate_descriptive_summary_tables()