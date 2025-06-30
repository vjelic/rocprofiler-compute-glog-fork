import yaml
import re
import itertools
import random
from collections import defaultdict

OUTPUT_FILE='src/rocprof_compute_soc/analysis_configs/gfx942/2300_test.yaml'
SUBSET_SIZE = 5

METRIC_FILES = [
    'src/rocprof_compute_soc/analysis_configs/gfx942/0200_system-speed-of-light.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/0300_mem_chart.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/0500_command-processor.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/0600_shader-processor-input.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1000_compute-unit-instruction-mix.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1100_compute-unit-compute-pipeline.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1200_lds.yaml'
]

IP_BLOCKS = {
    "SQ": 8,
    "TA": 2,
    "TD": 2,
    "TCP": 4,
    "TCC": 4,
    "CPC": 2,
    "CPF": 2,
    "SPI": 2,
    "GRBM": 2,
    "GDS": 4
}

IGNORE = [
    "2.1.0", # VALU FLOPs (uses >8 SQ counters)

    # Duplicates:
    "11.1.0", # VALU FLOPs
	"11.1.1", # VALU IOPs
	"11.1.2", # MFMA FLOPs (F8)
	"11.1.3", # MFMA FLOPs (BF16)
	"11.1.4", # MFMA FLOPs (F16)
	"11.1.5", # MFMA FLOPs (F32)
    "11.1.6", # MFMA FLOPs (F64)
    "11.1.7", # MFMA FLOPs (F6F4)
	"11.1.8", # MFMA IOPs (INT8)
    "11.2.0", # IPC
    "11.2.2", # SALU Utilization 
	"11.2.3", # VALU Utilization 
    "11.2.4", # VALU Co-Issue Efficiency 
    "11.2.5", # VMEM Utilization 
	"11.2.6", # Branch Utilization
	"11.2.7", # VALU Active Threads
	"11.2.8", # MFMA Utilization
	"11.2.9", # MFMA Instr Cycles
    "11.3.2", # F8 OPs
	"11.3.3", # F16 OPs
	"11.3.4", # BF16 OPs
	"11.3.5", # F32 OPs
	"11.3.6", # F64 OPs
	"11.3.7", # F6F4 OPs
	"11.3.8", # INT8 OPs
    "12.1.2", # Theoretical Bandwidth (% of Peak)
    "12.1.3", # Bank Conflict Rate
    "12.2.7", # Theoretical Bandwidth
    "12.2.9", # Bank Conflicts/Access
]

# For some reason rocprof-compute doesn't accept the output created
# by the yaml module, so we'll do it manually in the format that
# works with rocprof-compute
def write_metrics(metrics, id, title, file_path):

    # Bucket metrics according to their headers
    buckets = defaultdict(list)
    for m in metrics.keys():
        key = tuple(sorted(metrics[m]['header']))
        buckets[key].append(metrics[m])

    with open(file_path, 'wt') as f:
        print('Panel Config:', file=f)
        print(f'  id: {id * 100}', file=f)
        print(f"  title: {title}", file=f)
        print('  data source:', file=f)

        n = 1
        for k in buckets:
            
            # All metrics in bucket have same header
            header = buckets[k][0]['header']
            print('    - metric_table:', file=f)
            print(f'        id: {id * 100 + n}', file=f)
            print('        title: panel-test', file=f)
            print('        header:', file=f)
            for h in header:
                print(f'          {h}: {header[h]}', file=f)
            print('        metric:', file=f)
            for m in buckets[k]:
                print(m)
                print(f"          {m['name']}:", file=f)
                for h in k:
                    if h == 'metric':
                        continue
                    print(f"            {h}: {m[h]}", file=f)

            n += 1

def init_block_counts():
    block_counts = {}

    for b in IP_BLOCKS:
        block_counts[b] = 0
    
    return block_counts

def get_block_counts(counters):
    block_counts = init_block_counts()

    for ctr in counters:
        for b in IP_BLOCKS:
            if ctr.startswith(b):
                block_counts[b] += 1

    return block_counts

def load_metrics(counter_files):
    counter_info = {}
    for file_path in counter_files: 
        with open(file_path) as f:
            info = yaml.safe_load(f)

            sources = info['Panel Config']['data source']
            for source in sources:
                id = source['metric_table']['id']
                major = id // 100
                minor = id % 100
                ctrs = source['metric_table']['metric']
                header = source['metric_table']['header']

                counters_dict = {}
                k = 0
                for c in ctrs.keys():
                    m = ctrs[c].copy()
                    m['name'] = c
                    m['header'] = header
                    id_str = f'{major}.{minor}.{k}'
                    k += 1

                    if id_str in IGNORE:
                        continue
                    
                    counters_dict[id_str] = m

                counter_info.update(counters_dict)

    return counter_info

def write_pmc(metrics):
    counters = []

    for m in metrics:
        counters.extend(m['counters'])

    counters = list(set(counters))

    s = ' '.join(counters)

    pmc_str = f'pmc: {s}\n'
    with open('pmc.txt', 'wt') as r:
        r.write(pmc_str)




def get_available_metrics():
    available_metrics = []

    metrics = load_metrics(METRIC_FILES)
    
    # Convert the metrics dictionary into a list
    # Also determines what counters are required for each metric
    for id in metrics.keys():

        name = metrics[id]['name']

        value = metrics[id]['value'] if 'value' in metrics[id] else None
        avg = metrics[id]['avg'] if 'avg' in metrics[id] else None
        maximum = metrics[id]['max'] if 'max' in metrics[id] else None
        minimum = metrics[id]['min'] if 'min' in metrics[id] else None
        unit = metrics[id]['unit'] if 'unit' in metrics[id] else None
        peak = metrics[id]['peak'] if 'peak' in metrics[id] else None
        pop = metrics[id]['pop'] if 'pop' in metrics[id] else None
        
        header = metrics[id]['header']

        # Parse out the counter names
        if value != None:
            v = value
        elif avg != None:
            v = avg 
        else:
            print('Skipping {} {}'.format(id, name))
            continue
 
        parsed = re.split(r'[!=+\-/*() ]+', v)

        counters = []
        for p in parsed:

            if len(p) == 0:
                continue

            if p[0] == '$':
                if p == '$numActiveCUs':
                    counters.append('GRBM_GUI_ACTIVE')
                    counters.append('SQ_BUSY_CU_CYCLES')
                elif p == '$GRBM_GUI_ACTIVE_PER_XCD':
                    counters.append('GRBM_GUI_ACTIVE')
                continue

            for block in IP_BLOCKS:
                if block in p:
                    counters.append(p.strip())
                    break

        # Remove duplicates
        counters = list(set(counters))

        if len(counters) == 0:
            continue
     
        available_metrics.append({'id':id,
                                  'name':name,
                                  'counters':counters,
                                  'header':header,
                                  'value':value,
                                  'avg':avg,
                                  'unit':unit,
                                  'peak':peak,
                                  'pop':pop,
                                  'max':maximum,
                                  'min':minimum})

    return available_metrics


def get_combos(metrics_list, must_have, subset_size):

    m_copy = metrics_list.copy()

    metrics_must_have = []

    # Remove the metrics we know we want
    for m in m_copy[:]:
        if m['id'] in must_have:
            metrics_must_have.append(m)
            m_copy.remove(m)

    # Get the number of counters used by our "must have" metrics
    ctrs = []
    for m in metrics_must_have:
        ctrs.extend(m['counters'])
    
    ctrs = list(set(ctrs))

    ctr_counts = get_block_counts(ctrs)

    max_counts_local = IP_BLOCKS.copy()
    for c in ctr_counts.keys():
        max_counts_local[c] -= ctr_counts[c]

    combinations = itertools.combinations(m_copy, subset_size - len(must_have))

    final_list = []
    for combo in combinations:

        # Collect the counters
        ctrs = []
        for m in combo:
            ctrs.extend(m['counters'])

        # Remove duplicate counters
        ctrs = list(set(ctrs))
        counts = get_block_counts(ctrs)
        
        # Reject if goes beyond max count
        works = True
        for k in counts.keys():
            if counts[k] > max_counts_local[k]:
                works = False

        if not works:
            continue
        
        final_list.append(list(combo))

    # Add the must-have metrics back in
    for m in final_list:
        m.extend(metrics_must_have)

    return final_list

def print_selection_menu(metrics):

    cols = 3
    names = list(f"{m['name']}" for m in metrics)

    for i in range(0, len(names), cols):
        indices = list(range(i, min(i+3, len(names))))
        print("".join(f"{idx+1:>3}. {names[idx]:<40}" for idx in indices))
    
def print_counter_usage(metrics):

    counters = []
    for m in metrics:
        counters.extend(m['counters'])
    
    counters = list(set(counters))
    block_counts = get_block_counts(counters)

    print(''.join(f'{b}: {block_counts[b]}/{IP_BLOCKS[b]}  ' for b in IP_BLOCKS))

def select_metrics(metrics):

    selected_ids = set()

    while True:
        print_selection_menu(metrics)
        if len(selected_ids) > 0:
            print('Current selection: ')
            for id in selected_ids:
                print(f"  {metrics[id]['name']}")

        print('0. Done')
        print_counter_usage([metrics[id] for id in selected_ids])

        idx = int(input("Select: "))
        if idx == 0:
            break
        elif idx > 0:
            selected_ids.add(idx-1)
        else:
            if -idx-1 in selected_ids:
                selected_ids.remove(-idx-1)

    return [metrics[i]['id'] for i in selected_ids]

def main_interactive():
    metrics = get_available_metrics()

    selected = select_metrics(metrics)

    print('Selected {}'.format(selected))

    combos = get_combos(metrics, selected, SUBSET_SIZE)

    print('Found {} combinations'.format(len(combos)))

    if len(combos) == 0:
        print('No configurations found')
        exit(0)

    # Select random subset
    selected = list(combos[random.randint(0, len(combos))])

    print('Random selection:')    
    for m in selected:
        print(f"{m['id']} {m['name']}")
 
    dict = {}
    for m in selected:
        name = m['name']
        dict[name] = {'name':name,
                      'id':m['id'],
                      'value':m['value'],
                      'avg':m['avg'],
                      'header':m['header'],
                      'unit':m['unit'],
                      'peak':m['peak'],
                      'pop':None,
                      'tips':None,
                      'max':m['max'],
                      'min':m['min']}

    # Write to file
    write_metrics(dict, 23, 'Test Panel', OUTPUT_FILE)

def main():
    MUST_HAVE = [
        '2.1.9',
        '2.1.15',
        '2.1.17',
    ]

    metrics = get_available_metrics()

    print(f'Loaded {len(metrics)} metrics')

    combos = get_combos(metrics, MUST_HAVE, SUBSET_SIZE)

    print('Found {} combinations'.format(len(combos)))

    if len(combos) == 0:
        print('No configurations found')
        exit(0)

    # Select random subset
    selected = list(combos[random.randint(0, len(combos))])
    dict = {}
    for m in selected:
        name = m['name']
        dict[name] = {'name':name,
                      'id':m['id'],
                      'value':m['value'],
                      'avg':m['avg'],
                      'unit':m['unit'],
                      'peak':m['peak'],
                      'pop':None,
                      'tips':None,
                      'max':m['max'],
                      'min':m['min']}

    # Write to file
    write_metrics(dict, 23, 'Test Panel', OUTPUT_FILE)

if __name__ == "__main__":
    #main()
    main_interactive()
