# create_dataset.py
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

def read_signal(path):
    """Reads a signal text file and returns a Dataframe with datetime index and one value column."""
    data_started = False
    timestamps = []
    values = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "data:":
                data_started = True
                continue
            if not data_started or not line:
                continue
            try:
                time_str, val = line.split(";")
                val = float(val.strip())
                ts = pd.to_datetime(time_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
                timestamps.append(ts)
                values.append(val)
            except:
                continue

    if len(timestamps) == 0:
        print(f"Warning: No valid data in {path}")
        return pd.DataFrame(columns=["value"], index=pd.to_datetime([]))

    return pd.DataFrame({"value": values}, index=timestamps)


def read_events(path):
    """Reads FlowEvents file into a DataFrame."""
    data_started = False
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "data:":
                data_started = True
                continue
            if not data_started or not line:
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            try:
                start = pd.to_datetime(parts[0].strip(), format="%d.%m.%Y %H:%M:%S,%f")
                end = pd.to_datetime(parts[1].strip(), format="%d.%m.%Y %H:%M:%S,%f")
                event = parts[2].strip()
                rows.append([start, end, event])
            except:
                continue
    return pd.DataFrame(rows, columns=["start_time", "end_time", "event_type"])


def find_file(prefix, folder):
    """Find the first file starting with prefix in a folder."""
    for f in os.listdir(folder):
        if f.startswith(prefix):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{prefix} file not found in {folder}")


#window and labelling
WINDOW_SEC = 30
OVERLAP = 0.5
FS_FLOW = 32
FS_THORAC = 32
FS_SPO2 = 4

WINDOW_FLOW = WINDOW_SEC * FS_FLOW
STEP_FLOW = int(WINDOW_FLOW * (1 - OVERLAP))

WINDOW_THORAC = WINDOW_SEC * FS_THORAC
STEP_THORAC = int(WINDOW_THORAC * (1 - OVERLAP))

WINDOW_SPO2 = WINDOW_SEC * FS_SPO2
STEP_SPO2 = int(WINDOW_SPO2 * (1 - OVERLAP))


def label_window(start_time, end_time, events):
    """Label a window based on >50% overlap with events."""
    for _, e in events.iterrows():
        overlap_start = max(start_time, e['start_time'])
        overlap_end = min(end_time, e['end_time'])
        overlap_sec = (overlap_end - overlap_start).total_seconds()
        if overlap_sec / WINDOW_SEC > 0.5:
            if e['event_type'] in ["Hypopnea", "Obstructive Apnea"]:
                return e['event_type']
    return "Normal"


def create_windows(signal, window_samples, step_samples):
    """Split 1D signal into overlapping windows."""
    data = []
    idx = 0
    while idx + window_samples <= len(signal):
        data.append(signal[idx: idx + window_samples])
        idx += step_samples
    return np.array(data)


#processing of participants
def process_participant(participant_dir):
    participant = os.path.basename(participant_dir)
    print("Processing:", participant)

    try:
        flow_path = find_file("Flow", participant_dir)
        thorac_path = find_file("Thorac", participant_dir)
        spo2_path = find_file("SPO2", participant_dir)
        events_path = find_file("FlowEvents", participant_dir)
    except FileNotFoundError as e:
        print(f"Skipping {participant}: {e}")
        return None

    flow = read_signal(flow_path)
    thorac = read_signal(thorac_path)
    spo2 = read_signal(spo2_path)
    events = read_events(events_path)

    #skipping if any signal is empty
    if len(flow) == 0 or len(thorac) == 0 or len(spo2) == 0:
        print(f"Skipping {participant}: one of the signals is empty")
        return None

    #aligning signals in time
    min_time = max(flow.index[0], thorac.index[0], spo2.index[0])
    max_time = min(flow.index[-1], thorac.index[-1], spo2.index[-1])
    flow = flow[min_time:max_time]
    thorac = thorac[min_time:max_time]
    spo2 = spo2[min_time:max_time]

    flow_array = flow['value'].values
    thorac_array = thorac['value'].values
   #interpolation
    spo2_array = np.interp(
        np.linspace(0, len(spo2)-1, num=len(flow_array)),
        np.arange(len(spo2)),
        spo2['value'].values
    )

    #creating windows
    flow_windows = create_windows(flow_array, WINDOW_FLOW, STEP_FLOW)
    thorac_windows = create_windows(thorac_array, WINDOW_THORAC, STEP_THORAC)
    spo2_windows = create_windows(spo2_array, WINDOW_FLOW, STEP_FLOW)

    #generating labels
    timestamps = flow.index[::STEP_FLOW]
    labels = []
    for start_time in timestamps[:len(flow_windows)]:
        end_time = start_time + pd.Timedelta(seconds=WINDOW_SEC)
        lbl = label_window(start_time, end_time, events)
        labels.append(lbl)

    dataset = {
        "flow": flow_windows,
        "thorac": thorac_windows,
        "spo2": spo2_windows,
        "labels": labels
    }

    return dataset


#main function
def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    participants = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for p in participants:
        p_dir = os.path.join(data_dir, p)
        dataset = process_participant(p_dir)
        if dataset is None:
            continue
        out_path = os.path.join(out_dir, f"{p}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        print("Saved dataset for", p, "->", out_path)


#main program execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True, help="Input Data folder")
    parser.add_argument("-out_dir", required=True, help="Output Dataset folder")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)