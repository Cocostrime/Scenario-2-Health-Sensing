import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#reading signal files
def read_signal(path):
    data_started = False
    timestamps = []
    values = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "Data:":
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

    df = pd.DataFrame({"value": values}, index=timestamps)
    return df

#Reading relevant files
def read_events(path):
    data_started = False
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "Data:":
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

#File finding
def find_file(prefix, folder):
    for f in os.listdir(folder):
        if f.startswith(prefix):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{prefix} file not found in {folder}")

#Visualisation section
def main(participant_dir):
    participant = os.path.basename(participant_dir)

    # Finding all relevant files
    flow_path = find_file("Flow", participant_dir)
    thorac_path = find_file("Thorac", participant_dir)
    spo2_path = find_file("SPO2", participant_dir)
    events_path = find_file("FlowEvents", participant_dir)
    sleepprofile_path = find_file("SleepProfile", participant_dir)

    # Reading signals
    flow = read_signal(flow_path)
    thorac = read_signal(thorac_path)
    spo2 = read_signal(spo2_path)
    sleepprofile = read_signal(sleepprofile_path)
    events = read_events(events_path)

    # Creating the output folder
    os.makedirs("Visualizations", exist_ok=True)
    pdf_path = f"Visualizations/{participant}.pdf"

    # Plotting of all signals
    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        axs[0].plot(flow.index, flow["value"], color='blue')
        axs[0].set_title("Nasal Airflow")
        axs[0].set_ylabel("Flow")

        axs[1].plot(thorac.index, thorac["value"], color='green')
        axs[1].set_title("Thoracic Movement")
        axs[1].set_ylabel("Thorac")

        axs[2].plot(spo2.index, spo2["value"], color='red')
        axs[2].set_title("SpO₂")
        axs[2].set_ylabel("%")

        axs[3].plot(sleepprofile.index, sleepprofile["value"], color='purple')
        axs[3].set_title("Sleep Profile")
        axs[3].set_ylabel("Stage")

        # Overlay events
        for _, e in events.iterrows():
            for ax in axs:
                ax.axvspan(e["start_time"], e["end_time"], color='orange', alpha=0.3)

        plt.xlabel("Time")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print("Visualization saved:", pdf_path)

#main program execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", required=True, help="Participant folder path")
    args = parser.parse_args()
    main(args.name)