import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

isolated_dir = os.path.join(repo_root, "data", "custom_eval", "isolated")
dpi_path = os.path.join(isolated_dir, "drift_point_info.csv")
di_path = os.path.join(isolated_dir, "drift_info.csv")

dpi_df = pd.read_csv(dpi_path, sep=";")
print("Read drift_point_info.csv:")
print(dpi_df)

lines = ["log_name;drift_or_noise_id;drift_attribute;drift_sub_attribute;value"]

grouped = dpi_df.groupby("log_name", sort=False)

for log_name, group in grouped:
    for idx, row in enumerate(group.itertuples(), start=1):
        drift_id = f"drift_{idx}"
        cp = int(row.actual_drift_point)
        lines.append(f"{log_name};{drift_id};log_id;na;{log_name}")
        lines.append(f"{log_name};{drift_id};drift_id;na;{idx}")
        lines.append(f"{log_name};{drift_id};process_perspective;na;control-flow")
        lines.append(f"{log_name};{drift_id};drift_type;na;sudden")
        lines.append(f"{log_name};{drift_id};change_info_1;change_start;2024-05-01 00:00:00.000000 [{cp}]")
        lines.append(f"{log_name};{drift_id};change_info_1;change_end;")

with open(di_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSuccessfully generated {di_path} from {dpi_path}.")
