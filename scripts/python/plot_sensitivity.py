import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sensitivity.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print("CSV is empty, skipping plot.")
            sys.exit(0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Grab the lag value for the title
    lag_val = df['lag'].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df['window_size'], df['F1'], marker='o', linewidth=2, label='F1 Score')
    plt.plot(df['window_size'], df['Precision'], marker='s', linestyle='--', label='Precision')
    plt.plot(df['window_size'], df['Recall'], marker='^', linestyle='--', label='Recall')

    plt.title(f'Window Size Sensitivity Analysis (Lag = {lag_val})')
    plt.xlabel('Number of Windows (N_WINDOWS)')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    
    # Optional: Rotate x-ticks slightly if there are many sizes
    plt.xticks(df['window_size'], rotation=45)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plot_file = csv_file.replace('.csv', '.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Plot successfully saved to {plot_file}")

if __name__ == "__main__":
    main()
