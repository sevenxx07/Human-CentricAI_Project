import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

METRICS_DIR = "metrics_data"


def parse_metrics_file(filepath):
    data = {
        'session_id': None,
        'mode': None,
        'rmse': None,
        'num_ratings': None,
        'num_skips': None,
        'avg_time_per_rating': None
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Session ID"):
                data['session_id'] = line.split(":")[1].strip()
            elif line.startswith("Study Mode"):
                data['mode'] = line.split(":")[1].strip()
            elif line.startswith("RMSE"):
                match = re.search(r"RMSE:\s+([\d\.]+)", line)
                if match:
                    data['rmse'] = float(match.group(1))
            elif line.startswith("Number of Ratings"):
                data['num_ratings'] = int(line.split(":")[1].strip())
            elif line.startswith("Number of Skips"):
                data['num_skips'] = int(line.split(":")[1].strip())
            elif line.startswith("Average Time per Rating"):
                match = re.search(r"([\d\.]+)s", line)
                if match:
                    data['avg_time_per_rating'] = float(match.group(1))
    return data


def aggregate_results():
    results = []
    for file in os.listdir(METRICS_DIR):
        if file.endswith(".txt"):
            results.append(parse_metrics_file(os.path.join(METRICS_DIR, file)))

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = aggregate_results()
    print("Aggregated Results:")
    print(df)

    # Group by study mode
    grouped = df.groupby("mode").agg({
        'rmse': 'mean',
        'num_ratings': 'mean',
        'num_skips': 'mean',
        'avg_time_per_rating': 'mean'
    })
    print("\nGroup Statistics:\n", grouped)

    # T-test RMSE between guided and unguided
    guided_rmse = df[df['mode'] == 'guided']['rmse'].dropna()
    unguided_rmse = df[df['mode'] == 'unguided']['rmse'].dropna()

    if len(guided_rmse) > 1 and len(unguided_rmse) > 1:
        t_stat, p_val = ttest_ind(guided_rmse, unguided_rmse)
        print(f"\nT-test RMSE Guided vs Unguided: t={t_stat:.3f}, p={p_val:.3f}")

    # Visualization
    df.boxplot(column='rmse', by='mode')
    plt.title("RMSE by Study Mode")
    plt.ylabel("RMSE")
    plt.show()