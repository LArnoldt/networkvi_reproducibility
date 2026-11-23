import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


def load_benchmark_results(base_paths):
    '''Load benchmark results from multiple subsampling experiments'''
    all_results = []

    for sample_size, base_path in base_paths.items():
        results_file = Path(base_path) / "scaling_analysis_results.csv"

        if results_file.exists():
            df = pd.read_csv(results_file)
            df['sample_size'] = sample_size

            if sample_size == 'full':
                df['n_cells_numeric'] = df['n_cells'].iloc[0]
            elif 'k' in sample_size.lower():
                df['n_cells_numeric'] = int(sample_size.lower().replace('k', '')) * 1000
            elif 'm' in sample_size.lower():
                df['n_cells_numeric'] = int(sample_size.lower().replace('m', '')) * 1000000
            else:
                df['n_cells_numeric'] = df['n_cells']

            all_results.append(df)
        else:
            print(f"Warning: Results file not found at {results_file}")

    if not all_results:
        raise ValueError("No results files found!")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values('n_cells_numeric')

    return combined_df


def plot_training_time_vs_cells(df, output_dir, log_scale=True):
    """Plot training time vs number of cells"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline_df = df[df['experiment'].str.contains('baseline')]

    axes[0].plot(baseline_df['n_cells_numeric'], baseline_df['training_time_hours'],
                 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Training Time')
    axes[0].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if log_scale:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')

    if 'interpretability_time_hours' in df.columns:
        axes[1].plot(baseline_df['n_cells_numeric'],
                     baseline_df['interpretability_time_hours'],
                     'o-', linewidth=2, markersize=8, color='#A23B72',
                     label='GO Importance')

        if 'gpu_memory_covariate_attention_allocated_mb' in df.columns:
            total_interp = (baseline_df['interpretability_time_hours'].fillna(0) +
                            baseline_df.get('interpretability_time_hours', 0))
            axes[1].plot(baseline_df['n_cells_numeric'], total_interp,
                         's--', linewidth=2, markersize=8, color='#F18F01',
                         label='Total Interpretability')

    axes[1].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Interpretability Time (hours)', fontsize=12, fontweight='bold')
    axes[1].set_title('Interpretability Time vs Dataset Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    if log_scale:
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'time_scaling.png', bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'time_scaling.pdf', bbox_inches='tight')
    plt.close()


def plot_memory_vs_cells(df, output_dir, log_scale=True):
    """Plot memory usage vs number of cells"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline_df = df[df['experiment'].str.contains('baseline')]

    axes[0, 0].plot(baseline_df['n_cells_numeric'],
                    baseline_df['gpu_memory_training_allocated_mb'],
                    'o-', linewidth=2, markersize=8, color='#06A77D', label='Allocated')
    if 'gpu_memory_training_reserved_mb' in baseline_df.columns:
        axes[0, 0].plot(baseline_df['n_cells_numeric'],
                        baseline_df['gpu_memory_training_reserved_mb'],
                        's--', linewidth=2, markersize=8, color='#005E7C', label='Reserved')
    axes[0, 0].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('GPU Memory - Training', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    if log_scale:
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')

    if 'gpu_memory_go_importance_allocated_mb' in baseline_df.columns:
        axes[0, 1].plot(baseline_df['n_cells_numeric'],
                        baseline_df['gpu_memory_go_importance_allocated_mb'],
                        'o-', linewidth=2, markersize=8, color='#D62828', label='GO Importance')
        if 'gpu_memory_covariate_attention_allocated_mb' in baseline_df.columns:
            axes[0, 1].plot(baseline_df['n_cells_numeric'],
                            baseline_df['gpu_memory_covariate_attention_allocated_mb'],
                            's-', linewidth=2, markersize=8, color='#F77F00',
                            label='Covariate Attention')
    axes[0, 1].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('GPU Memory - Interpretability', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    if log_scale:
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')

    axes[1, 0].plot(baseline_df['n_cells_numeric'],
                    baseline_df['cpu_memory_training_mb'],
                    'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[1, 0].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('CPU Memory (MB)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('CPU Memory - Training', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    if log_scale:
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')

    if 'cpu_memory_go_importance_mb' in baseline_df.columns:
        axes[1, 1].plot(baseline_df['n_cells_numeric'],
                        baseline_df['cpu_memory_go_importance_mb'],
                        'o-', linewidth=2, markersize=8, color='#A23B72', label='GO Importance')
        if 'cpu_memory_covariate_attention_mb' in baseline_df.columns:
            axes[1, 1].plot(baseline_df['n_cells_numeric'],
                            baseline_df['cpu_memory_covariate_attention_mb'],
                            's-', linewidth=2, markersize=8, color='#F18F01',
                            label='Covariate Attention')
    axes[1, 1].set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('CPU Memory (MB)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('CPU Memory - Interpretability', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    if log_scale:
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'memory_scaling.png', bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'memory_scaling.pdf', bbox_inches='tight')
    plt.close()


def plot_parameter_effects(df, output_dir):
    """Plot the effect of varying parameters on performance"""

    sample_sizes = df['sample_size'].unique()

    for sample_size in sample_sizes:
        sample_df = df[df['sample_size'] == sample_size].copy()

        sample_df['param_varied'] = sample_df['experiment'].apply(
            lambda x: x.split('/')[-1] if '/' in x else 'baseline'
        )

        baseline = sample_df[sample_df['param_varied'] == 'baseline']
        variations = sample_df[sample_df['param_varied'] != 'baseline']

        if len(variations) == 0:
            continue

        variations['param_name'] = variations['param_varied'].apply(
            lambda x: x.split('_')[0] if '_' in x else x
        )
        variations['param_value'] = variations['param_varied'].apply(
            lambda x: int(x.split('_')[1]) if '_' in x else None
        )

        varied_params = variations['param_name'].unique()

        n_params = len(varied_params)
        fig, axes = plt.subplots(n_params, 3, figsize=(15, 5 * n_params))

        if n_params == 1:
            axes = axes.reshape(1, -1)

        for idx, param in enumerate(varied_params):
            param_df = variations[variations['param_name'] == param].sort_values('param_value')

            axes[idx, 0].plot(param_df['param_value'], param_df['training_time_hours'],
                              'o-', linewidth=2, markersize=8, color='#2E86AB')
            axes[idx, 0].axhline(y=baseline['training_time_hours'].values[0],
                                 color='red', linestyle='--', alpha=0.5, label='Baseline')
            axes[idx, 0].set_xlabel(f'{param}', fontsize=12, fontweight='bold')
            axes[idx, 0].set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
            axes[idx, 0].set_title(f'Training Time vs {param}\n({sample_size} cells)',
                                   fontsize=12, fontweight='bold')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            axes[idx, 1].plot(param_df['param_value'],
                              param_df['gpu_memory_training_allocated_mb'],
                              'o-', linewidth=2, markersize=8, color='#06A77D')
            axes[idx, 1].axhline(y=baseline['gpu_memory_training_allocated_mb'].values[0],
                                 color='red', linestyle='--', alpha=0.5, label='Baseline')
            axes[idx, 1].set_xlabel(f'{param}', fontsize=12, fontweight='bold')
            axes[idx, 1].set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
            axes[idx, 1].set_title(f'GPU Memory vs {param}\n({sample_size} cells)',
                                   fontsize=12, fontweight='bold')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

            axes[idx, 2].plot(param_df['param_value'],
                              param_df['cpu_memory_training_mb'],
                              'o-', linewidth=2, markersize=8, color='#A23B72')
            axes[idx, 2].axhline(y=baseline['cpu_memory_training_mb'].values[0],
                                 color='red', linestyle='--', alpha=0.5, label='Baseline')
            axes[idx, 2].set_xlabel(f'{param}', fontsize=12, fontweight='bold')
            axes[idx, 2].set_ylabel('CPU Memory (MB)', fontsize=12, fontweight='bold')
            axes[idx, 2].set_title(f'CPU Memory vs {param}\n({sample_size} cells)',
                                   fontsize=12, fontweight='bold')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'parameter_effects_{sample_size}.png',
                    bbox_inches='tight')
        plt.savefig(Path(output_dir) / f'parameter_effects_{sample_size}.pdf',
                    bbox_inches='tight')
        plt.close()


def plot_combined_summary(df, output_dir):
    """Create a comprehensive summary figure"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    baseline_df = df[df['experiment'].str.contains('baseline')].sort_values('n_cells_numeric')

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(baseline_df['n_cells_numeric'], baseline_df['training_time_hours'],
               'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of Cells', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Training Time (hours)', fontsize=10, fontweight='bold')
    ax1.set_title('Training Time Scaling', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(baseline_df['n_cells_numeric'],
               baseline_df['gpu_memory_training_allocated_mb'],
               'o-', linewidth=2, markersize=8, color='#06A77D')
    ax2.set_xlabel('Number of Cells', fontsize=10, fontweight='bold')
    ax2.set_ylabel('GPU Memory (MB)', fontsize=10, fontweight='bold')
    ax2.set_title('GPU Memory Scaling', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.loglog(baseline_df['n_cells_numeric'],
               baseline_df['cpu_memory_training_mb'],
               'o-', linewidth=2, markersize=8, color='#A23B72')
    ax3.set_xlabel('Number of Cells', fontsize=10, fontweight='bold')
    ax3.set_ylabel('CPU Memory (MB)', fontsize=10, fontweight='bold')
    ax3.set_title('CPU Memory Scaling', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(baseline_df))
    width = 0.35

    ax4.bar(x - width / 2, baseline_df['training_time_hours'], width,
            label='Training', color='#2E86AB', alpha=0.8)
    if 'interpretability_time_hours' in baseline_df.columns:
        ax4.bar(x + width / 2, baseline_df['interpretability_time_hours'], width,
                label='Interpretability', color='#A23B72', alpha=0.8)

    ax4.set_xlabel('Dataset Size', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Time (hours)', fontsize=10, fontweight='bold')
    ax4.set_title('Training vs Interpretability Time', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(baseline_df['sample_size'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = fig.add_subplot(gs[1, 2])
    memory_data = []
    labels = []

    for idx, row in baseline_df.iterrows():
        memory_data.append([
            row['gpu_memory_training_allocated_mb'],
            row.get('gpu_memory_go_importance_allocated_mb', 0)
        ])
        labels.append(row['sample_size'])

    memory_array = np.array(memory_data)
    x_pos = np.arange(len(labels))

    ax5.bar(x_pos, memory_array[:, 0], label='Training', color='#06A77D', alpha=0.8)
    ax5.bar(x_pos, memory_array[:, 1], bottom=memory_array[:, 0],
            label='Interpretability', color='#D62828', alpha=0.8)

    ax5.set_xlabel('Dataset Size', fontsize=10, fontweight='bold')
    ax5.set_ylabel('GPU Memory (MB)', fontsize=10, fontweight='bold')
    ax5.set_title('GPU Memory Breakdown', fontsize=11, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    for idx, row in baseline_df.iterrows():
        table_data.append([
            row['sample_size'],
            f"{row['n_cells']:,.0f}",
            f"{row['training_time_hours']:.2f}",
            f"{row['gpu_memory_training_allocated_mb']:.0f}",
            f"{row['cpu_memory_training_mb']:.0f}",
            f"{row.get('interpretability_time_hours', 0):.2f}",
            f"{row.get('n_go_terms', 'N/A')}"
        ])

    table = ax6.table(cellText=table_data,
                      colLabels=['Sample', 'Cells', 'Train (h)', 'GPU (MB)',
                                 'CPU (MB)', 'Interp (h)', 'GO Terms'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.12, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(7):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.savefig(Path(output_dir) / 'comprehensive_summary.png', bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'comprehensive_summary.pdf', bbox_inches='tight')
    plt.close()


def generate_all_plots(base_paths, output_dir):
    '''Generate all benchmark plots'''
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading benchmark results...")
    df = load_benchmark_results(base_paths)

    df.to_csv(Path(output_dir) / 'combined_benchmark_results.csv', index=False)
    print(f"Combined results saved to {output_dir}/combined_benchmark_results.csv")

    print("\nGenerating plots...")

    print("  - Time scaling plots...")
    plot_training_time_vs_cells(df, output_dir)

    print("  - Memory scaling plots...")
    plot_memory_vs_cells(df, output_dir)

    print("  - Parameter effect plots...")
    plot_parameter_effects(df, output_dir)

    print("  - Comprehensive summary...")
    plot_combined_summary(df, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("\nGenerated files:")
    for file in Path(output_dir).glob('*.png'):
        print(f"  - {file.name}")


if __name__ == "__main__":
    base_paths = {
        '10k': '/path/to/hlca_10k_sparsevi_ml_benchmark',
        '100k': '/path/to/hlca_100k_sparsevi_ml_benchmark',
        '1M': '/path/to/hlca_1m_sparsevi_ml_benchmark',
        '2M': '/path/to/hlca_2m_sparsevi_ml_benchmark',
        'full': '/path/to/hlca_full_sparsevi_ml_benchmark',
    }

    output_dir = '/path/to/benchmark_plots'

    generate_all_plots(base_paths, output_dir)