import yaml
import sys
import json
import re
import itertools

# Max counters for each GFX IP block:
#"SQ": 8
#"TA": 2
#"TD": 2
#"TCP": 4
#"TCC": 4
#"CPC": 2
#"CPF": 2
#"SPI": 2
#"GRBM": 2
#"GDS": 4

OUTPUT_FILE='src/rocprof_compute_soc/analysis_configs/gfx942/2300_test.yaml'
SUBSET_SIZE = 5

max_counts = {
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

ip_blocks = [
    "SQ",
    "TA",
    "TD",
    "TCP", 
    "TCC",
    "CPC",
    "CPF",
    "SPI",
    "GRBM",
    "GDS"
]

to_remove = [
    #'VALU FLOPs',
    #  'vL1D Cache Hit Rate',
    #  'L1I Hit Rate',
    #  'L1I Fetch Latency',
    #  'Branch Utilization',
    #  'Theoretical LDS Bandwidth',
    #  'sL1D Cache BW',
    #  'L2-Fabric Read Latency',
    #  'L2-Fabric Write Latency',
    #  'L1I BW'
]

must_have = [
            'Active CUs',
            'Wavefront Occupancy',
            'VALU Active Threads',
]

def write_metrics(metrics, id, title, file_path):

    with open(file_path, 'wt') as f:
        print('Panel Config:', file=f)
        print(f'  id: {id * 100}', file=f)
        print(f"  title: {title}", file=f)
        print('  data source:', file=f)
        print('    - metric_table:', file=f)
        print(f'        id: {id * 100 + 1}', file=f)
        print('        title: panel-test', file=f)
        print('        header:', file=f)
        print('          metric: Metric', file=f)
        print('          value: Avg', file=f)
        print('          unit: Unit', file=f)
        print('          peak: Peak', file=f)
        print('          pop: Pct of Peak', file=f)
        print('          tips: Tips', file=f)
        print('        metric:', file=f)
        for k in metrics.keys():
            m = metrics[k]
            print(f"          {k}:", file=f)
            print(f"            value: {m['value']}", file=f)
            print(f"            unit: {m['unit']}", file=f)
            print(f"            peak: {m['peak']}", file=f)
            print(f"            pop: {m['pop']}", file=f)
            print(f'            tips:', file=f)

def init_block_counts():
    block_counts = {}

    for b in ip_blocks:
        block_counts[b] = 0
    
    return block_counts

def get_block_counts(counters):
    block_counts = init_block_counts()

    for ctr in counters:
        for b in ip_blocks:
            if ctr.startswith(b):
                block_counts[b] += 1

    return block_counts

def load_metrics(counter_files):
    counter_info = {}
    for file_path in counter_files: 
        with open(file_path) as f:
            info = yaml.safe_load(f)
            counter_info.update(info['Panel Config']['data source'][0]['metric_table']['metric'])

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

metric_files = [
    'src/rocprof_compute_soc/analysis_configs/gfx942/0200_system-speed-of-light.yaml',
    #'src/rocprof_compute_soc/analysis_configs/gfx942/0300_mem_chart.yaml'
    #'src/rocprof_compute_soc/analysis_configs/gfx942/0500_command-processor.yaml'
    #'src/rocprof_compute_soc/analysis_configs/gfx942/0600_shader-processor-input.yaml'
    #'src/rocprof_compute_soc/analysis_configs/gfx942/1000_compute-unit-instruction-mix.yaml'
    #'src/rocprof_compute_soc/analysis_configs/gfx942/1100_compute-unit-compute-pipeline.yaml'
    #'src/rocprof_compute_soc/analysis_configs/gfx942/1200_lds.yaml'
]

metrics = load_metrics(metric_files)
m_list = []

# Convert the metrics dictionary into a list
# Also determines what counters are required for each metric
for m in metrics.keys():

    # Some have value some have average
    if 'value' not in metrics[m]:
        if 'avg' not in metrics[m]:
            continue
        v = metrics[m]['avg']
    else:
        v = metrics[m]['value']

    if v == None:
        continue

    parsed = re.split(r'[!=+\-/*() ]+', v)

    unit = metrics[m]['unit'] if 'unit' in metrics[m] else None
    peak = metrics[m]['peak'] if 'peak' in metrics[m] else None
    pop = metrics[m]['pop'] if 'pop' in metrics[m] else None

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

        for block in ip_blocks:
            if block in p:
                counters.append(p.strip())
                break

    # Remove duplicates
    counters = list(set(counters))

    if len(counters) == 0:
        continue
 
    m_list.append({'name':m, 'counters':counters, 'value':v, 'unit':unit, 'peak':peak, 'pop':pop})



# Get the metric names
metric_names = [m['name'] for m in m_list]

# Fiter out the metric names we don't want
for m in metric_names[:]:
    if m in to_remove:
        metric_names.remove(m)

# Remaining names are the metrics we do want
m_list = [m for m in m_list if m.get('name') in metric_names]


# Every combination of 5 metrics
combinations = itertools.combinations(m_list, SUBSET_SIZE)


final_list = []
count = 0
for combo in combinations:
    count += 1


    # Reject if it does not have our must-have metrics
    works = True 
    for m in must_have:
        if m not in [c['name'] for c in combo]:
            works = False
    
    if not works:
        continue

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
        if counts[k] > max_counts[k]:
            works = False

    if not works:
        continue
    
    final_list.append(combo)

print('Rejectd {}'.format(count - len(final_list)))
print('Found {} combinations'.format(len(final_list)))

if len(final_list) == 0:
    print('No configurations found')
    exit(0)

metrics_dict = {}

metrics = list(final_list[0])

for m in metrics:
    name = m['name']
    metrics_dict[name] = {'value':m['value'], 'unit':m['unit'], 'peak':m['peak'], 'pop':None, 'tips':None}

write_metrics(metrics_dict, 23, 'Test Panel', OUTPUT_FILE)
#write_pmc(metrics)