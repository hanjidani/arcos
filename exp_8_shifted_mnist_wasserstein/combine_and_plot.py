import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

def combine_and_plot():
    """
    Combines all individual result CSVs into a single dataframe
    and generates heatmaps for the tightness ratio.
    """
    results_dir = os.path.join('exp_7_shifted_mnist', 'results')
    plots_dir = os.path.join('exp_7_shifted_mnist', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Find and combine all result files
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not all_files:
        print("No result files found. Did you run the Slurm jobs?")
        return

    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Save the combined dataframe
    combined_path = os.path.join(results_dir, 'all_results_combined.csv')
    df.to_csv(combined_path, index=False)
    print(f"Combined results saved to {combined_path}")

    # 2. Generate Heatmaps
    for shift_type in df['shift_type'].unique():
        print(f"\nGenerating heatmap for shift type: {shift_type}")
        
        # Filter for the current shift type
        df_type = df[df['shift_type'] == shift_type]
        
        # Create a pivot table for the heatmap
        pivot_df = df_type.pivot_table(
            index='num_layers', 
            columns='shift_severity', 
            values='tightness_ratio'
        )
        
        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis_r") # _r reverses the colormap
        
        plt.title(f'Bound Tightness Ratio (Bound / |ΔR|) for {shift_type.capitalize()} Shift')
        plt.xlabel('Shift Severity')
        plt.ylabel('Model Capacity (Number of Layers)')
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f'{shift_type}_tightness_heatmap.png')
        plt.savefig(plot_path)
        print(f"Heatmap saved to {plot_path}")
        plt.close()

if __name__ == '__main__':
    combine_and_plot()
