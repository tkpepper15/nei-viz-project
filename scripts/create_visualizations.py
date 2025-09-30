#!/usr/bin/env python3

"""
Visualization Generation Script for Circuit Analysis

Creates correlation heatmaps, PCA plots, and other visualizations
from the circuit analysis CSV data using matplotlib, seaborn, and plotly.

Usage: python scripts/create_visualizations.py [output_directory]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import argparse

if TYPE_CHECKING:
    from scripts.circuit_analysis import CircuitAnalyzer
from datetime import datetime

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Interactive plots will be skipped.")
    PLOTLY_AVAILABLE = False

# Scientific computing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr

class CircuitVisualizationGenerator:
    """Main class for generating circuit analysis visualizations"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV data files"""
        csv_dir = self.output_dir / "csv"
        data = {}

        # Load parameters data
        params_file = csv_dir / "parameters.csv"
        if params_file.exists():
            data['parameters'] = pd.read_csv(params_file)
            print(f"   Loaded {len(data['parameters'])} parameter records")
        else:
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        # Load spectra data
        spectra_file = csv_dir / "spectra.csv"
        if spectra_file.exists():
            data['spectra'] = pd.read_csv(spectra_file)
            print(f"   Loaded {len(data['spectra'])} spectrum points")

        # Load PCA directions data
        pca_file = csv_dir / "pca_directions.csv"
        if pca_file.exists():
            data['pca_directions'] = pd.read_csv(pca_file)
            print(f"   Loaded {len(data['pca_directions'])} PCA direction records")

        return data

    def create_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Create parameter correlation heatmap"""
        print("   📊 Creating correlation heatmap...")

        # Select numeric parameters for correlation
        param_cols = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb', 'resnorm', 'condition_number']
        corr_data = df[param_cols].corr()

        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))

        sns.heatmap(corr_data,
                    mask=mask,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.5},
                    ax=ax)

        ax.set_title('Circuit Parameter Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.viz_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create interactive Plotly version if available
        if PLOTLY_AVAILABLE:
            fig_plotly = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation Coefficient")
            ))

            fig_plotly.update_layout(
                title="Interactive Circuit Parameter Correlation Matrix",
                xaxis_title="Parameters",
                yaxis_title="Parameters",
                width=700,
                height=600
            )

            fig_plotly.write_html(self.viz_dir / "correlation_heatmap_interactive.html")

    def create_pca_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA analysis and create visualizations"""
        print("   📈 Creating PCA analysis...")

        # Prepare data for PCA
        param_cols = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']
        X = df[param_cols].values

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Create PCA results summary
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'feature_names': param_cols,
            'transformed_data': X_pca
        }

        # 1. Explained Variance Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scree plot
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_ * 100)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance (%)')
        ax1.set_title('PCA Scree Plot')
        ax1.grid(True, alpha=0.3)

        # Cumulative variance
        ax2.plot(range(1, len(pca_results['cumulative_variance']) + 1),
                 pca_results['cumulative_variance'] * 100, 'bo-')
        ax2.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% variance')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_explained_variance.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. PCA Score Plot
        fig, ax = plt.subplots(figsize=(12, 9))

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                           c=df['resnorm'],
                           cmap='viridis',
                           alpha=0.7,
                           s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title('PCA Score Plot (colored by resnorm)')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Resnorm')

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_score_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Component Loadings Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i in range(min(4, len(pca.components_))):
            ax = axes[i]
            loadings = pca.components_[i]

            bars = ax.bar(param_cols, np.abs(loadings))
            ax.set_title(f'PC{i+1} Loadings (|values|)')
            ax.set_ylabel('Absolute Loading')
            ax.tick_params(axis='x', rotation=45)

            # Color bars by sign
            for j, bar in enumerate(bars):
                if loadings[j] > 0:
                    bar.set_color('steelblue')
                else:
                    bar.set_color('orangered')

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_loadings.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create interactive PCA plot if Plotly is available
        if PLOTLY_AVAILABLE:
            fig_plotly = go.Figure()

            fig_plotly.add_trace(go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['resnorm'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Resnorm")
                ),
                text=[f"Variation {i}<br>Resnorm: {resnorm:.6f}"
                      for i, resnorm in enumerate(df['resnorm'])],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
            ))

            fig_plotly.update_layout(
                title=f"Interactive PCA Score Plot<br><sub>PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}% variance explained</sub>",
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                width=800,
                height=600
            )

            fig_plotly.write_html(self.viz_dir / "pca_interactive.html")

        return pca_results

    def create_pairs_plot(self, df: pd.DataFrame) -> None:
        """Create parameter pairs plot matrix"""
        print("   🔗 Creating pairs plot...")

        param_cols = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']

        # Create pairs plot
        g = sns.PairGrid(df[param_cols + ['resnorm']],
                        diag_sharey=False,
                        corner=True)

        # Upper triangle: scatter plots
        g.map_upper(sns.scatterplot, alpha=0.6, s=30)

        # Diagonal: histograms
        g.map_diag(sns.histplot, kde=True)

        # Lower triangle: scatter with regression line
        g.map_lower(sns.scatterplot, alpha=0.6, s=30)
        g.map_lower(sns.regplot, scatter=False, color='red', line_kws={'alpha': 0.7})

        # Adjust layout
        g.fig.suptitle('Circuit Parameters Pairs Plot', y=1.02, fontsize=16)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pairs_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_sensitivity_analysis(self, df: pd.DataFrame) -> None:
        """Create sensitivity analysis plots"""
        print("   🎯 Creating sensitivity analysis...")

        if 'pc1_sensitivity' not in df.columns:
            print("     Skipping sensitivity analysis - data not available")
            return

        # Sensitivity comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PC sensitivities
        ax = axes[0, 0]
        pc_cols = [col for col in df.columns if 'pc' in col and 'sensitivity' in col]
        if pc_cols:
            pc_data = df[pc_cols].mean()
            ax.bar(range(len(pc_data)), pc_data.values)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Mean Sensitivity')
            ax.set_title('Principal Component Sensitivities')
            ax.set_xticks(range(len(pc_data)))
            ax.set_xticklabels([f'PC{i+1}' for i in range(len(pc_data))])

        # Condition number distribution (filter out infinite values)
        ax = axes[0, 1]
        finite_condition_numbers = df['condition_number'][np.isfinite(df['condition_number'])]

        if len(finite_condition_numbers) > 0:
            ax.hist(finite_condition_numbers, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Condition Number')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Condition Number Distribution\n({len(finite_condition_numbers)}/{len(df)} finite values)')
            median_val = finite_condition_numbers.median()
            ax.axvline(median_val, color='red',
                      linestyle='--', label=f'Median: {median_val:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'All condition numbers\nare infinite',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Condition Number Distribution')

        # Resnorm vs Condition Number (filter infinite condition numbers)
        ax = axes[1, 0]
        finite_mask = np.isfinite(df['condition_number'])

        if finite_mask.sum() > 0:
            finite_df = df[finite_mask]
            scatter = ax.scatter(finite_df['resnorm'], finite_df['condition_number'],
                               alpha=0.6, c=finite_df['pc1_sensitivity'], cmap='plasma')
            ax.set_xlabel('Resnorm')
            ax.set_ylabel('Condition Number')
            ax.set_title(f'Resnorm vs Condition Number\n({finite_mask.sum()}/{len(df)} finite values)')
            ax.set_yscale('log')
            plt.colorbar(scatter, ax=ax, label='PC1 Sensitivity')
        else:
            ax.text(0.5, 0.5, 'All condition numbers\nare infinite',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Resnorm vs Condition Number')

        # Parameter space coverage
        ax = axes[1, 1]
        param_ranges = {}
        for param in ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']:
            param_ranges[param] = (df[param].max() - df[param].min()) / df[param].mean()

        bars = ax.bar(param_ranges.keys(), param_ranges.values())
        ax.set_ylabel('Relative Range (range/mean)')
        ax.set_title('Parameter Space Coverage')
        ax.tick_params(axis='x', rotation=45)

        # Color bars by magnitude
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))

        plt.tight_layout()
        plt.savefig(self.viz_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_spectrum_analysis(self, spectra_df: pd.DataFrame, params_df: pd.DataFrame) -> None:
        """Create spectrum analysis plots"""
        print("   🌊 Creating spectrum analysis...")

        if spectra_df is None or len(spectra_df) == 0:
            print("     Skipping spectrum analysis - data not available")
            return

        # Get top 5 and bottom 5 resnorm variations
        best_variations = params_df.nsmallest(5, 'resnorm')['variation_id'].values
        worst_variations = params_df.nlargest(5, 'resnorm')['variation_id'].values

        # 1. Nyquist plots comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Best fits
        for var_id in best_variations:
            var_data = spectra_df[spectra_df['variation_id'] == var_id]
            if len(var_data) > 0:
                ax1.plot(var_data['real'], -var_data['imaginary'],
                        'o-', markersize=3, alpha=0.7, linewidth=1)

        ax1.set_xlabel('Real Impedance (Ω)')
        ax1.set_ylabel('-Imaginary Impedance (Ω)')
        ax1.set_title('Nyquist Plots - Best Fits (lowest resnorm)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')

        # Worst fits
        for var_id in worst_variations:
            var_data = spectra_df[spectra_df['variation_id'] == var_id]
            if len(var_data) > 0:
                ax2.plot(var_data['real'], -var_data['imaginary'],
                        'o-', markersize=3, alpha=0.7, linewidth=1)

        ax2.set_xlabel('Real Impedance (Ω)')
        ax2.set_ylabel('-Imaginary Impedance (Ω)')
        ax2.set_title('Nyquist Plots - Worst Fits (highest resnorm)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(self.viz_dir / "nyquist_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Frequency response analysis
        # Sample a few variations for frequency response
        sample_variations = params_df.sample(min(10, len(params_df)))['variation_id'].values

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        for var_id in sample_variations:
            var_data = spectra_df[spectra_df['variation_id'] == var_id]
            if len(var_data) > 0:
                resnorm = params_df[params_df['variation_id'] == var_id]['resnorm'].iloc[0]

                # Magnitude vs frequency
                ax1.loglog(var_data['frequency'], var_data['magnitude'],
                          alpha=0.6, label=f'{var_id} (resnorm: {resnorm:.4f})')

                # Phase vs frequency
                ax2.semilogx(var_data['frequency'], var_data['phase'],
                           alpha=0.6)

                # Real part vs frequency
                ax3.loglog(var_data['frequency'], var_data['real'],
                          alpha=0.6)

                # Imaginary part vs frequency
                ax4.loglog(var_data['frequency'], -var_data['imaginary'],
                          alpha=0.6)

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('|Z| (Ω)')
        ax1.set_title('Impedance Magnitude vs Frequency')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Phase vs Frequency')
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Real(Z) (Ω)')
        ax3.set_title('Real Impedance vs Frequency')
        ax3.grid(True, alpha=0.3)

        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('-Imag(Z) (Ω)')
        ax4.set_title('Imaginary Impedance vs Frequency')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "frequency_response.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_jacobian_analysis(self, analyzer: Any, params_df: pd.DataFrame) -> None:
        """Create Jacobian matrix visualizations for circuit perturbations"""
        print("   🔬 Creating Jacobian analysis...")

        if not hasattr(analyzer, 'detailed_results') or len(analyzer.detailed_results) == 0:
            print("     Skipping Jacobian analysis - detailed results not available")
            return

        # Create Jacobian visualizations directory
        jacobian_dir = self.viz_dir / "jacobian_matrices"
        jacobian_dir.mkdir(exist_ok=True)

        # Get a representative sample of variations for visualization
        sample_size = min(12, len(analyzer.detailed_results))

        # Select variations to cover the resnorm range
        sorted_results = sorted(analyzer.detailed_results,
                              key=lambda x: params_df[params_df['variation_id'] == x['variation_id']]['resnorm'].iloc[0])

        # Sample evenly across the resnorm range
        indices = np.linspace(0, len(sorted_results)-1, sample_size, dtype=int)
        sample_results = [sorted_results[i] for i in indices]

        param_names = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']

        # 1. Individual Jacobian matrices for sampled variations
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()

        for i, detail in enumerate(sample_results):
            if i >= 12:  # Limit to 12 subplots
                break

            ax = axes[i]
            jacobian_real = detail['jacobian']['real']
            jacobian_imag = detail['jacobian']['imag']

            # Combine real and imaginary parts for visualization
            # Stack them vertically: [Real_part; Imag_part]
            jacobian_combined = np.vstack([jacobian_real, jacobian_imag])

            # Create frequency labels for y-axis
            frequencies = detail['frequencies']
            freq_labels = [f"{f:.1f}" for f in frequencies[:5]] + ["..."] + [f"{f:.1f}" for f in frequencies[-5:]]
            y_labels = [f"Re({fl})" for fl in freq_labels] + [f"Im({fl})" for fl in freq_labels]

            # Plot heatmap
            im = ax.imshow(jacobian_combined, cmap='RdBu_r', aspect='auto')

            # Get resnorm for this variation
            resnorm = params_df[params_df['variation_id'] == detail['variation_id']]['resnorm'].iloc[0]
            ax.set_title(f"{detail['variation_id']}\nResnorm: {resnorm:.4f}", fontsize=10)

            # Set parameter names on x-axis
            ax.set_xticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45)

            # Simplified y-axis (too many frequency points to show all)
            n_freq_show = min(5, len(frequencies))
            y_ticks = list(range(0, n_freq_show)) + list(range(len(frequencies), len(frequencies) + n_freq_show))
            y_tick_labels = [f"Re(f{i})" for i in range(n_freq_show)] + [f"Im(f{i})" for i in range(n_freq_show)]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=8)

            # Add colorbar for first plot only
            if i == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='∂Z/∂param')

        # Hide unused subplots
        for j in range(len(sample_results), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Jacobian Matrices Across Circuit Perturbations\n(Real and Imaginary Parts Stacked)',
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(jacobian_dir / "jacobian_grid.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Jacobian magnitude analysis across all variations
        self._create_jacobian_magnitude_analysis(analyzer, params_df, jacobian_dir, param_names)

        # 3. Frequency-resolved sensitivity analysis
        self._create_frequency_sensitivity_analysis(analyzer, params_df, jacobian_dir, param_names)

    def _create_jacobian_magnitude_analysis(self, analyzer, params_df: pd.DataFrame,
                                          jacobian_dir: Path, param_names: list) -> None:
        """Create analysis of Jacobian magnitudes across variations"""

        # Collect Jacobian magnitudes for all variations
        jacobian_magnitudes = []
        resnorms = []
        variation_ids = []

        for detail in analyzer.detailed_results:
            jacobian_real = detail['jacobian']['real']
            jacobian_imag = detail['jacobian']['imag']

            # Calculate magnitude of complex Jacobian
            jacobian_mag = np.sqrt(jacobian_real**2 + jacobian_imag**2)

            # Average magnitude across frequencies for each parameter
            avg_magnitudes = np.mean(jacobian_mag, axis=0)

            jacobian_magnitudes.append(avg_magnitudes)
            resnorm = params_df[params_df['variation_id'] == detail['variation_id']]['resnorm'].iloc[0]
            resnorms.append(resnorm)
            variation_ids.append(detail['variation_id'])

        jacobian_magnitudes = np.array(jacobian_magnitudes)
        resnorms = np.array(resnorms)

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Jacobian magnitude heatmap
        im1 = ax1.imshow(jacobian_magnitudes.T, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Variation Index')
        ax1.set_ylabel('Parameters')
        ax1.set_title('Average Jacobian Magnitudes Across Variations')
        ax1.set_yticks(range(len(param_names)))
        ax1.set_yticklabels(param_names)
        plt.colorbar(im1, ax=ax1, label='|∂Z/∂param| (avg)')

        # 2. Parameter sensitivity vs resnorm
        for i, param in enumerate(param_names):
            ax2.scatter(resnorms, jacobian_magnitudes[:, i], alpha=0.6, label=param, s=30)
        ax2.set_xlabel('Resnorm')
        ax2.set_ylabel('Average Jacobian Magnitude')
        ax2.set_title('Parameter Sensitivity vs Circuit Performance')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Parameter sensitivity distribution
        ax3.boxplot([jacobian_magnitudes[:, i] for i in range(len(param_names))],
                   labels=param_names)
        ax3.set_ylabel('Average Jacobian Magnitude')
        ax3.set_title('Parameter Sensitivity Distribution')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Sensitivity correlation matrix
        sensitivity_corr = np.corrcoef(jacobian_magnitudes.T)
        im4 = ax4.imshow(sensitivity_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Parameter Sensitivity Correlations')
        ax4.set_xticks(range(len(param_names)))
        ax4.set_xticklabels(param_names, rotation=45)
        ax4.set_yticks(range(len(param_names)))
        ax4.set_yticklabels(param_names)
        plt.colorbar(im4, ax=ax4, label='Correlation')

        # Add correlation values as text
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                ax4.text(j, i, f'{sensitivity_corr[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(jacobian_dir / "jacobian_magnitude_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_frequency_sensitivity_analysis(self, analyzer, params_df: pd.DataFrame,
                                             jacobian_dir: Path, param_names: list) -> None:
        """Create frequency-resolved sensitivity analysis"""

        # Select a few representative variations
        sample_size = min(5, len(analyzer.detailed_results))
        sorted_results = sorted(analyzer.detailed_results,
                              key=lambda x: params_df[params_df['variation_id'] == x['variation_id']]['resnorm'].iloc[0])

        indices = np.linspace(0, len(sorted_results)-1, sample_size, dtype=int)
        sample_results = [sorted_results[i] for i in indices]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot sensitivity vs frequency for each parameter
        for param_idx, param_name in enumerate(param_names):
            ax = axes[param_idx // 3, param_idx % 3]

            for detail in sample_results:
                jacobian_real = detail['jacobian']['real']
                jacobian_imag = detail['jacobian']['imag']
                frequencies = detail['frequencies']

                # Calculate magnitude of Jacobian for this parameter
                param_sensitivity = np.sqrt(jacobian_real[:, param_idx]**2 +
                                          jacobian_imag[:, param_idx]**2)

                resnorm = params_df[params_df['variation_id'] == detail['variation_id']]['resnorm'].iloc[0]

                ax.loglog(frequencies, param_sensitivity,
                         alpha=0.7, linewidth=2,
                         label=f"Resnorm: {resnorm:.4f}")

            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'|∂Z/∂{param_name}|')
            ax.set_title(f'Frequency Sensitivity: {param_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide the 6th subplot (we only have 5 parameters)
        axes[1, 2].set_visible(False)

        plt.suptitle('Parameter Sensitivity vs Frequency Across Circuit Variations',
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(jacobian_dir / "frequency_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create interactive Plotly version if available
        if PLOTLY_AVAILABLE:
            self._create_interactive_jacobian_plot(sample_results, params_df, jacobian_dir, param_names)

    def _create_interactive_jacobian_plot(self, sample_results: list, params_df: pd.DataFrame,
                                        jacobian_dir: Path, param_names: list) -> None:
        """Create interactive Jacobian visualization with Plotly"""

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'∂Z/∂{param}' for param in param_names],
            specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*2 + [None]]
        )

        colors = px.colors.qualitative.Set1

        for detail_idx, detail in enumerate(sample_results):
            jacobian_real = detail['jacobian']['real']
            jacobian_imag = detail['jacobian']['imag']
            frequencies = detail['frequencies']

            resnorm = params_df[params_df['variation_id'] == detail['variation_id']]['resnorm'].iloc[0]
            color = colors[detail_idx % len(colors)]

            for param_idx, param_name in enumerate(param_names):
                param_sensitivity = np.sqrt(jacobian_real[:, param_idx]**2 +
                                          jacobian_imag[:, param_idx]**2)

                row = (param_idx // 3) + 1
                col = (param_idx % 3) + 1

                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=param_sensitivity,
                        mode='lines',
                        name=f'{detail["variation_id"]} (resnorm: {resnorm:.4f})',
                        line=dict(color=color),
                        showlegend=(param_idx == 0),  # Only show legend for first parameter
                        hovertemplate=f"<b>{param_name}</b><br>" +
                                    "Frequency: %{x:.1f} Hz<br>" +
                                    "Sensitivity: %{y:.2e}<br>" +
                                    f"Resnorm: {resnorm:.4f}<extra></extra>"
                    ),
                    row=row, col=col
                )

        # Update layout
        fig.update_layout(
            title_text="Interactive Parameter Sensitivity vs Frequency",
            height=800,
            showlegend=True
        )

        # Update axes to log scale
        for i in range(1, 6):  # 5 parameters
            row = ((i-1) // 3) + 1
            col = ((i-1) % 3) + 1
            fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=row, col=col)
            fig.update_yaxes(type="log", title_text="Sensitivity", row=row, col=col)

        fig.write_html(jacobian_dir / "interactive_jacobian_analysis.html")

    def create_summary_report(self, params_df: pd.DataFrame, pca_results: Dict[str, Any]) -> None:
        """Create comprehensive summary report"""
        print("   📋 Creating summary report...")

        # Calculate summary statistics
        stats = {
            'n_variations': len(params_df),
            'resnorm_range': (params_df['resnorm'].min(), params_df['resnorm'].max()),
            'resnorm_mean': params_df['resnorm'].mean(),
            'condition_number_median': params_df['condition_number'].median(),
            'pc1_variance': pca_results['explained_variance_ratio'][0] * 100,
            'pc2_variance': pca_results['explained_variance_ratio'][1] * 100,
            'cumulative_pc12': (pca_results['explained_variance_ratio'][0] +
                               pca_results['explained_variance_ratio'][1]) * 100
        }

        # Find most variable parameter
        param_cols = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']
        cv_values = {}
        for param in param_cols:
            cv = params_df[param].std() / params_df[param].mean()
            cv_values[param] = cv
        most_variable = max(cv_values, key=cv_values.get)

        # Find strongest correlations
        corr_matrix = params_df[param_cols + ['resnorm']].corr()
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.1:  # Only significant correlations
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        corr_pairs.sort(key=lambda x: x[2], reverse=True)

        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Circuit Analysis Summary Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: #f8f9fa; margin: 20px 0; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2.2em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #6c757d; font-size: 0.9em; margin-top: 5px; }}
        .visualization-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .viz-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .viz-card img {{ width: 100%; height: auto; border-radius: 4px; }}
        h1, h2 {{ color: #333; }}
        .findings ul {{ list-style-type: none; padding: 0; }}
        .findings li {{ background: white; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 3px solid #28a745; }}
        .correlation-list {{ background: white; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Circuit Parameter Analysis Summary Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analysis of {stats['n_variations']} circuit parameter variations</p>
    </div>

    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-value">{stats['n_variations']}</div>
            <div class="stat-label">Parameter Variations</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{stats['pc1_variance']:.1f}%</div>
            <div class="stat-label">PC1 Variance Explained</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{stats['pc2_variance']:.1f}%</div>
            <div class="stat-label">PC2 Variance Explained</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{stats['cumulative_pc12']:.1f}%</div>
            <div class="stat-label">Cumulative PC1+PC2</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{stats['resnorm_mean']:.4f}</div>
            <div class="stat-label">Mean Resnorm</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{stats['condition_number_median']:.1f}</div>
            <div class="stat-label">Median Condition Number</div>
        </div>
    </div>

    <div class="section findings">
        <h2>🔍 Key Findings</h2>
        <ul>
            <li><strong>Most Variable Parameter:</strong> {most_variable} (CV: {cv_values[most_variable]*100:.1f}%)</li>
            <li><strong>Resnorm Range:</strong> {stats['resnorm_range'][0]:.6f} - {stats['resnorm_range'][1]:.6f}</li>
            <li><strong>Parameter Identifiability:</strong> {"Good" if stats['condition_number_median'] < 10 else "Moderate" if stats['condition_number_median'] < 100 else "Poor"} (median condition number: {stats['condition_number_median']:.1f})</li>
            <li><strong>PCA Effectiveness:</strong> First two components explain {stats['cumulative_pc12']:.1f}% of variance</li>
        </ul>
    </div>

    <div class="section">
        <h2>🔗 Strongest Parameter Correlations</h2>
        <div class="correlation-list">
            {"".join([f"<p><strong>{pair[0]} ↔ {pair[1]}:</strong> r = {pair[2]:.3f}</p>" for pair in corr_pairs[:5]])}
        </div>
    </div>

    <div class="section">
        <h2>📊 Generated Visualizations</h2>
        <div class="visualization-grid">
            <div class="viz-card">
                <h3>Correlation Heatmap</h3>
                <img src="correlation_heatmap.png" alt="Correlation Heatmap">
                <p>Parameter correlation matrix showing linear relationships between circuit elements.</p>
            </div>
            <div class="viz-card">
                <h3>PCA Analysis</h3>
                <img src="pca_score_plot.png" alt="PCA Score Plot">
                <p>Principal component analysis revealing parameter space structure.</p>
            </div>
            <div class="viz-card">
                <h3>Sensitivity Analysis</h3>
                <img src="sensitivity_analysis.png" alt="Sensitivity Analysis">
                <p>Parameter sensitivity and condition number distributions.</p>
            </div>
            <div class="viz-card">
                <h3>Pairs Plot</h3>
                <img src="pairs_plot.png" alt="Pairs Plot">
                <p>Pairwise parameter relationships with correlation trends.</p>
            </div>
            <div class="viz-card">
                <h3>Jacobian Analysis</h3>
                <img src="jacobian_matrices/jacobian_grid.png" alt="Jacobian Matrices">
                <p>Parameter sensitivity matrices showing directional derivatives ∂Z/∂param.</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>💾 Data Files</h2>
        <ul>
            <li><strong>parameters.csv:</strong> Circuit parameters, resnorm values, and condition numbers</li>
            <li><strong>spectra.csv:</strong> Complete impedance spectra for each parameter variation</li>
            <li><strong>pca_directions.csv:</strong> Principal component directions and sensitivities</li>
            <li><strong>analysis_config.json:</strong> Configuration used for this analysis</li>
        </ul>
    </div>

    <div class="section">
        <h2>🎯 Interpretation Guide</h2>
        <h3>Correlation Coefficients:</h3>
        <ul>
            <li><strong>|r| > 0.7:</strong> Strong correlation - parameters are highly related</li>
            <li><strong>0.3 < |r| < 0.7:</strong> Moderate correlation - noticeable relationship</li>
            <li><strong>|r| < 0.3:</strong> Weak correlation - little linear relationship</li>
        </ul>

        <h3>Condition Numbers:</h3>
        <ul>
            <li><strong>< 10:</strong> Well-conditioned - parameters easily distinguishable</li>
            <li><strong>10-100:</strong> Moderately conditioned - some parameter ambiguity</li>
            <li><strong>> 100:</strong> Ill-conditioned - difficult parameter estimation</li>
        </ul>
    </div>
</body>
</html>
        """

        # Save HTML report
        with open(self.viz_dir / "summary_report.html", 'w') as f:
            f.write(html_content)

    def generate_all_visualizations(self) -> None:
        """Generate complete set of visualizations"""
        print("\n📊 Creating visualizations...")

        # Load data
        print("📥 Loading data...")
        data = self.load_data()
        params_df = data['parameters']
        spectra_df = data.get('spectra')
        pca_directions_df = data.get('pca_directions')

        # Generate visualizations
        self.create_correlation_heatmap(params_df)
        pca_results = self.create_pca_analysis(params_df)
        self.create_pairs_plot(params_df)
        self.create_sensitivity_analysis(params_df)

        if spectra_df is not None:
            self.create_spectrum_analysis(spectra_df, params_df)

        # Create Jacobian analysis (pass the analyzer object)
        if hasattr(self, '_analyzer_ref'):
            self.create_jacobian_analysis(self._analyzer_ref, params_df)

        self.create_summary_report(params_df, pca_results)

        print(f"\n🎉 All visualizations created in: {self.viz_dir}")
        print("\n📋 Generated files:")
        for file_path in sorted(self.viz_dir.glob("*")):
            print(f"   📄 {file_path.name}")

def find_recent_analysis_dirs() -> list:
    """Find recent analysis output directories"""
    analysis_dir = Path("./analysis_output")
    if not analysis_dir.exists():
        return []

    dirs = [d for d in analysis_dir.iterdir()
            if d.is_dir() and (d / "csv" / "parameters.csv").exists()]

    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate visualizations for circuit analysis")
    parser.add_argument("output_dir", nargs="?", help="Analysis output directory")
    parser.add_argument("--list", action="store_true", help="List available analysis directories")
    args = parser.parse_args()

    print("📊 Circuit Analysis Visualization Generator")
    print("=" * 50)

    # Handle listing recent directories
    if args.list:
        recent_dirs = find_recent_analysis_dirs()
        if recent_dirs:
            print("\nAvailable analysis directories:")
            for i, dir_path in enumerate(recent_dirs[:10], 1):
                print(f"  {i}. {dir_path.name}")
        else:
            print("\nNo analysis directories found.")
        return

    # Determine output directory
    output_dir = None

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"❌ Directory not found: {output_dir}")
            return
    else:
        # Interactive selection
        recent_dirs = find_recent_analysis_dirs()
        if not recent_dirs:
            print("❌ No analysis directories found.")
            print("   Run: python scripts/circuit_analysis.py first")
            return

        print("\nRecent analysis directories:")
        for i, dir_path in enumerate(recent_dirs[:5], 1):
            mtime = datetime.fromtimestamp(dir_path.stat().st_mtime)
            print(f"  {i}. {dir_path.name} (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")

        try:
            choice = input(f"\nSelect directory (1-{min(5, len(recent_dirs))}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(recent_dirs):
                output_dir = recent_dirs[choice_idx]
            else:
                print("❌ Invalid selection")
                return
        except (ValueError, KeyboardInterrupt):
            print("\n⏹️ Cancelled")
            return

    # Verify required files exist
    csv_dir = output_dir / "csv"
    params_file = csv_dir / "parameters.csv"

    if not params_file.exists():
        print(f"❌ Required file not found: {params_file}")
        print("   Make sure you've run the circuit analysis script first.")
        return

    try:
        # Generate visualizations
        viz_generator = CircuitVisualizationGenerator(str(output_dir))

        # Create a dummy analyzer reference for accessing detailed results from saved data
        # In a real scenario, you'd load the analyzer state or recreate it
        class DummyAnalyzer:
            def __init__(self, detailed_results):
                self.detailed_results = detailed_results

        # Load detailed results if available
        detailed_results = []
        try:
            import pickle
            detailed_results_file = output_dir / "analysis" / "detailed_results.pkl"
            if detailed_results_file.exists():
                with open(detailed_results_file, 'rb') as f:
                    detailed_results = pickle.load(f)
        except:
            pass  # Continue without detailed results if loading fails

        # Set analyzer reference for Jacobian analysis
        if detailed_results:
            viz_generator._analyzer_ref = DummyAnalyzer(detailed_results)

        viz_generator.generate_all_visualizations()

        print(f"\n🌐 Open visualizations in your browser:")
        print(f"   📊 Summary Report: {viz_generator.viz_dir / 'summary_report.html'}")
        if PLOTLY_AVAILABLE:
            print(f"   🔗 Interactive Plots: {viz_generator.viz_dir / 'correlation_heatmap_interactive.html'}")
            print(f"   📈 Interactive PCA: {viz_generator.viz_dir / 'pca_interactive.html'}")

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        raise

if __name__ == "__main__":
    main()