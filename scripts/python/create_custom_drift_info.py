import os
import glob
import re
import pandas as pd
from dateutil import parser
import pm4py
from tqdm import tqdm

def process_dataset(dataset_dir):
    drift_csv = os.path.join(dataset_dir, "drift_info.csv")
    if not os.path.exists(drift_csv):
        return
        
    print(f"\nProcessing {dataset_dir}...")
    df = pd.read_csv(drift_csv, sep=';')
    
    # We will iterate through each log file that appears in drift_info
    log_names = df['log_name'].unique()
    
    for log_name in tqdm(log_names):
        log_path = os.path.join(dataset_dir, log_name)
        if not os.path.exists(log_path):
            continue
            
        # Parse XML manually to find timestamps
        drift_starts = {}
        drift_ends = {}
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "drift info" in line:
                    match = re.search(r'drift info (\d+)', line)
                    if not match:
                        continue
                    drift_id = match.group(1)
                    
                    start_match = re.search(r'drift start timestamp:\s*([^;]+)', line)
                    if start_match:
                        ts_str = start_match.group(1).split('(')[0].strip()
                        if ts_str != 'N/A':
                            drift_starts[drift_id] = parser.parse(ts_str)
                            
                    end_match = re.search(r'drift end timestamp:\s*([^;"]+)', line)
                    if end_match:
                        ts_str = end_match.group(1).split('(')[0].strip()
                        if ts_str != 'N/A':
                            drift_ends[drift_id] = parser.parse(ts_str)
                # Break early if we reach traces
                if "<trace>" in line:
                    break
                    
        if not drift_starts:
            continue
            
        # Map timestamps to trace index
        log = pm4py.read_xes(log_path)
        traces = log.groupby('case:concept:name', sort=False) if isinstance(log, pd.DataFrame) else log
        
        start_indices = {}
        end_indices = {}
        
        trace_idx = 0
        for case_id, trace in traces if isinstance(log, pd.DataFrame) else enumerate(traces):
            if isinstance(log, pd.DataFrame):
                start_time = trace['time:timestamp'].min()
            else:
                # get timestamp of first event
                start_time = trace[0]['time:timestamp']
                
            for d_id, ts in drift_starts.items():
                if d_id not in start_indices:
                    # align tz
                    cmp_ts = ts
                    if start_time.tzinfo is not None and cmp_ts.tzinfo is None:
                        cmp_ts = cmp_ts.replace(tzinfo=start_time.tzinfo)
                    if start_time >= cmp_ts:
                        start_indices[d_id] = trace_idx
                        
            for d_id, ts in drift_ends.items():
                if d_id not in end_indices:
                    cmp_ts = ts
                    if start_time.tzinfo is not None and cmp_ts.tzinfo is None:
                        cmp_ts = cmp_ts.replace(tzinfo=start_time.tzinfo)
                    if start_time >= cmp_ts:
                        end_indices[d_id] = trace_idx
            
            trace_idx += 1
            
        # Update the dataframe
        for d_id, st_idx in start_indices.items():
            # Match rows for change_start
            mask = (df['log_name'] == log_name) & (df['drift_or_noise_id'] == f'drift_{d_id}') & (df['drift_sub_attribute'] == 'change_start')
            for idx, row in df[mask].iterrows():
                val = row['value']
                # Replace (0.x) with [st_idx]
                new_val = re.sub(r'\([^\)]+\)', f'[{st_idx}]', str(val))
                df.at[idx, 'value'] = new_val
                
        for d_id, en_idx in end_indices.items():
            mask = (df['log_name'] == log_name) & (df['drift_or_noise_id'] == f'drift_{d_id}') & (df['drift_sub_attribute'] == 'change_end')
            for idx, row in df[mask].iterrows():
                val = row['value']
                new_val = re.sub(r'\([^\)]+\)', f'[{en_idx}]', str(val))
                df.at[idx, 'value'] = new_val
                
    # Save back
    df.to_csv(drift_csv, sep=';', index=False)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "custom_eval"))
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):
            process_dataset(full_path)
            
if __name__ == "__main__":
    main()
