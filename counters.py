import yaml
import sys
import json
import re
import itertools
import random

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
SUBSET_SIZE = 6

MAX_COUNTS = {
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

# For some reason rocprof-compute doesn't accept the output created
# by the yaml module, so we'll do it manually in the format that
# works with rocprof-compute
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
    'src/rocprof_compute_soc/analysis_configs/gfx942/0300_mem_chart.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/0500_command-processor.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/0600_shader-processor-input.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1000_compute-unit-instruction-mix.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1100_compute-unit-compute-pipeline.yaml',
    'src/rocprof_compute_soc/analysis_configs/gfx942/1200_lds.yaml'
]

def get_available_metrics(to_omit):
    available_metrics = []

    metrics = load_metrics(metric_files)
    
    # Convert the metrics dictionary into a list
    # Also determines what counters are required for each metric
    for m in metrics.keys():

        # Remove metrics we do not want
        if m in to_omit:
            continue

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
     
        available_metrics.append({'name':m, 'counters':counters, 'value':v, 'unit':unit, 'peak':peak, 'pop':pop})

    return available_metrics

m_list = get_available_metrics(to_remove)
print(f'Loaded {len(m_list)} metrics')


def get_combos(metrics_list, must_have, subset_size):

    m_copy = metrics_list.copy()

    metrics_must_have = []

    # Remove the metrics we know we want
    for m in m_copy[:]:
        if m['name'] in must_have:
            metrics_must_have.append(m)
            m_copy.remove(m)

    # Get the number of counters used by our "must have" metrics
    ctrs = []
    for m in metrics_must_have:
        ctrs.extend(m['counters'])
    
    ctrs = list(set(ctrs))

    ctr_counts = get_block_counts(ctrs)

    max_counts_local = MAX_COUNTS.copy()
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

final_list = get_combos(m_list, must_have, SUBSET_SIZE)

#print('Rejectd {}'.format(count - len(final_list)))
print('Found {} combinations'.format(len(final_list)))

if len(final_list) == 0:
    print('No configurations found')
    exit(0)

metrics_dict = {}

metrics = list(final_list[random.randint(0, len(final_list))])

for m in metrics:
    name = m['name']
    metrics_dict[name] = {'value':m['value'], 'unit':m['unit'], 'peak':m['peak'], 'pop':None, 'tips':None}

write_metrics(metrics_dict, 23, 'Test Panel', OUTPUT_FILE)
#write_pmc(metrics)

