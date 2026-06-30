import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
from sklearn.metrics import mean_squared_error
from get_data import get_data
import matplotlib
import pandas as pd
from scipy.signal import hilbert

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class CEEMDANAnalyzer:
    def __init__(self, signal, dates=None):
        self.signal = np.array(signal)
        self.dates = dates if dates is not None else np.arange(len(signal))
        self.imfs = None
        self.best_params = None
        self.best_score = None
        self.convergence_curve = None

    def objective_function(self, params):
        noise_std, ensemble_size = params
        noise_std = max(noise_std, 0.01)  # Noise std lower bound
        ensemble_size = int(max(ensemble_size, 10))  # Ensemble size lower bound

        ceemdan = CEEMDAN(noise_std=noise_std, ensemble_size=ensemble_size)
        imfs = ceemdan(self.signal)

        # Reconstruct signal (excluding residual if present)
        reconstructed = np.sum(imfs[:-1], axis=0) if len(imfs) > 1 else imfs[0]

        return mean_squared_error(self.signal, reconstructed)

    def grey_wolf_optimizer(self, lb, ub, search_agents=5, max_iter=10):
        dim = len(lb)
        # Initialize population
        positions = np.random.uniform(lb, ub, (search_agents, dim))
        alpha_pos = np.zeros(dim)
        beta_pos = np.zeros(dim)
        delta_pos = np.zeros(dim)
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        # Store convergence
        convergence_curve = np.zeros(max_iter)

        # Optimization loop
        for iter_idx in range(max_iter):
            for i in range(search_agents):
                # Calculate fitness
                fitness = self.objective_function(positions[i])

                # Update alpha, beta, delta
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[i].copy()

            # Update parameter a (linearly decreasing)
            a = 2 - iter_idx * (2 / max_iter)

            # Update positions
            for i in range(search_agents):
                for j in range(dim):
                    # Generate random numbers for each dimension
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    # Update position
                    positions[i, j] = (X1 + X2 + X3) / 3

                    # Enforce bounds
                    positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])

            convergence_curve[iter_idx] = alpha_score
            print(f"Iteration {iter_idx + 1}/{max_iter}: Best MSE = {alpha_score:.6f}")

        self.best_params = alpha_pos
        self.best_score = alpha_score
        self.convergence_curve = convergence_curve

        return alpha_pos, alpha_score, convergence_curve

    def optimize_and_decompose(self, lb, ub, search_agents=5, max_iter=10):
        """
        Optimize CEEMDAN parameters and perform decomposition

        Parameters:
        lb : list
            Lower bounds for [noise_std, ensemble_size]
        ub : list
            Upper bounds for [noise_std, ensemble_size]
        search_agents : int, optional
            Number of wolves in the population
        max_iter : int, optional
            Maximum number of iterations

        Returns:
        array: IMFs from CEEMDAN decomposition
        """
        # Run optimization
        self.grey_wolf_optimizer(lb, ub, search_agents, max_iter)

        # Extract optimal parameters
        noise_std_opt = self.best_params[0]
        ensemble_size_opt = int(self.best_params[1])

        print(f"\nOptimization Results:")
        print(f"Optimal Noise Std: {noise_std_opt:.4f}")
        print(f"Optimal Ensemble Size: {ensemble_size_opt}")
        print(f"Minimum Reconstruction MSE: {self.best_score:.6f}")

        # Perform CEEMDAN with optimal parameters
        ceemdan = CEEMDAN(noise_std=noise_std_opt, ensemble_size=ensemble_size_opt)
        self.imfs = ceemdan(self.signal)

        return self.imfs

    def plot_ceemdan_results(self):
        """Plot the CEEMDAN decomposition results"""
        if self.imfs is None:
            raise ValueError("IMFs not computed. Run optimize_and_decompose() first.")

        plt.figure(figsize=(10, 12))
        n_imfs = len(self.imfs)
        print('IMF count:', n_imfs)

        t = np.arange(len(self.signal))

        # Original signal
        plt.subplot(n_imfs + 1, 1, 1)
        plt.plot(t, self.signal, 'r')
        plt.title("Original Signal")
        plt.grid(True)

        # IMFs
        for i, imf in enumerate(self.imfs):
            plt.subplot(n_imfs + 1, 1, i + 2)
            plt.plot(t, imf)
            plt.title(f"IMF {i + 1}" if i < n_imfs - 1 else "Residual")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Reconstructed signal
        reconstructed = np.sum(self.imfs[:], axis=0) if n_imfs > 1 else self.imfs[0]
        plt.figure(figsize=(10, 6))
        plt.plot(t, self.signal, 'r', label='Original')
        plt.plot(t, reconstructed, 'b--', linewidth=1.5, label='Reconstructed')
        plt.title('Original vs Reconstructed Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Reconstruction error
        reconstruction_error = self.signal - reconstructed
        print(f"Reconstruction MSE: {mean_squared_error(self.signal, reconstructed):.6f}")

        plt.figure(figsize=(10, 2))
        plt.plot(t, reconstruction_error, 'g')
        plt.title('Reconstruction Error')
        plt.grid(True)
        plt.show()

    def plot_convergence(self):
        """Plot the optimization convergence curve"""
        if self.convergence_curve is None:
            raise ValueError("No convergence data. Run optimization first.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'b-', linewidth=2)
        plt.title('GWO Optimization Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best MSE')
        plt.grid(True)
        plt.show()

    @staticmethod
    def compute_instantaneous_period(imf, fs=1):
        """
        Calculate instantaneous period of an IMF

        Parameters:
        imf : array-like
            Intrinsic Mode Function
        fs : int, optional
            Sampling frequency (default=1 for daily data)

        Returns:
        array: Instantaneous period values
        """
        analytic_signal = hilbert(imf)
        phase = np.unwrap(np.angle(analytic_signal))

        # Calculate instantaneous frequency (central difference method)
        freq = np.zeros_like(phase)
        freq[1:-1] = (phase[2:] - phase[:-2]) * fs / (4 * np.pi)
        freq[0] = (phase[1] - phase[0]) * fs / (2 * np.pi)
        freq[-1] = (phase[-1] - phase[-2]) * fs / (2 * np.pi)

        # Calculate instantaneous period and filter invalid values
        period = np.full_like(freq, np.nan)
        valid_mask = np.abs(freq) > 1e-6
        period[valid_mask] = 1.0 / np.abs(freq[valid_mask])

        return period

    def plot_period_analysis(self, title="IMF Period Analysis", log_scale=True):
        """
        Plot instantaneous periods over time and period distributions

        Parameters:
        title : str, optional
            Title for the analysis
        log_scale : bool, optional
            Whether to use logarithmic scale for period analysis
        """
        if self.imfs is None:
            raise ValueError("IMFs not computed. Run optimize_and_decompose() first.")

        n_imfs = len(self.imfs)

        # Calculate instantaneous periods
        all_periods = [self.compute_instantaneous_period(imf) for imf in self.imfs]

        # Create figure
        plt.figure(figsize=(14, 10) if not log_scale else (10, 8))

        # Colors for IMFs
        colors = plt.cm.viridis(np.linspace(0, 1, n_imfs))

        # 1. Instantaneous Periods Over Time
        ax1 = plt.subplot(2, 1, 1)

        for i in range(n_imfs):
            valid_idx = ~np.isnan(all_periods[i])
            if np.any(valid_idx):
                ax1.plot(self.dates[valid_idx],
                         all_periods[i][valid_idx],
                         color=colors[i],
                         linewidth=1.5,
                         label=f'IMF{i + 1}')

        ax1.set_title(f'Instantaneous Periods Over Time - {title}', fontsize=14)
        ax1.set_ylabel('Period (days)', fontsize=12)
        if log_scale:
            ax1.set_yscale('log')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='best', ncol=min(4, n_imfs), fontsize=10)

        # Add reference periods
        ref_periods = [(7, '1 week'), (30, '1 month'), (90, '3 months'),
                       (180, '6 months'), (365, '1 year')]
        for days, label in ref_periods:
            ax1.axhline(y=days, color='gray', linestyle='--', alpha=0.5)
            ax1.text(self.dates[-1], days * (1.05 if log_scale else 1.05),
                     label, fontsize=9, verticalalignment='bottom',
                     backgroundcolor='white')

        # 2. Period Distributions
        ax2 = plt.subplot(2, 1, 2)

        # Set period range
        min_period = 1
        max_period = 365 * 3 if log_scale else 730  # 3 years for log, 2 years for linear

        # Create bins
        bins = (np.logspace(np.log10(min_period), np.log10(max_period), 50)
                if log_scale else
                np.linspace(0, max_period, 50))

        for i in range(n_imfs):
            valid_periods = all_periods[i][~np.isnan(all_periods[i])]
            valid_periods = valid_periods[(valid_periods >= min_period) &
                                          (valid_periods <= max_period)]

            if len(valid_periods) > 0:
                hist, bin_edges = np.histogram(valid_periods, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                if log_scale:
                    ax2.plot(bin_centers, hist,
                             color=colors[i],
                             linewidth=2,
                             label=f'IMF{i + 1}')
                else:
                    ax2.bar(bin_centers, hist,
                            width=np.diff(bin_edges)[0],
                            color=colors[i],
                            alpha=0.7,
                            edgecolor='k',
                            linewidth=0.5)

                # Add median line
                median_period = np.median(valid_periods)
                ax2.axvline(median_period, color=colors[i], linestyle='--', alpha=0.7)
                ax2.text(median_period * (1.05 if log_scale else 1.05),
                         np.max(hist) * 0.8,
                         f'Median: {median_period:.1f}d',
                         fontsize=9, color=colors[i])

        ax2.set_title('Period Distributions' + (' (Log Scale)' if log_scale else ''),
                      fontsize=14)
        ax2.set_xlabel('Period (days)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        if log_scale:
            ax2.set_xscale('log')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='best', fontsize=10)

        # Add reference periods
        for days, label in [(7, '1w'), (30, '1m'), (90, '3m'), (180, '6m'), (365, '1y')]:
            ax2.axvline(x=days, color='gray', linestyle='--', alpha=0.3)
            pos = days * (0.95 if log_scale else 1.0)
            ax2.text(pos, ax2.get_ylim()[1] * 0.9, label,
                     fontsize=8,
                     horizontalalignment='right' if log_scale else 'left')

        plt.tight_layout()
        plt.savefig(f'{title.replace(" ", "_")}_period_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Detailed period distributions for each IMF
        plt.figure(figsize=(14, 4 * n_imfs) if not log_scale else (10, 2 * n_imfs))

        # Set detailed view range
        detailed_max = 365 if not log_scale else max_period

        for i in range(n_imfs):
            ax = plt.subplot(n_imfs, 1, i + 1)
            valid_periods = all_periods[i][~np.isnan(all_periods[i])]
            valid_periods = valid_periods[(valid_periods >= min_period) &
                                          (valid_periods <= detailed_max)]

            if len(valid_periods) > 0:
                # Create bins for detailed view
                detailed_bins = (np.logspace(np.log10(min_period), np.log10(detailed_max), 30)
                                 if log_scale else
                                 np.linspace(0, detailed_max, 30))

                hist, bin_edges = np.histogram(valid_periods, bins=detailed_bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.bar(bin_centers, hist,
                       width=np.diff(bin_edges),
                       color=colors[i],
                       alpha=0.7,
                       edgecolor='k',
                       linewidth=0.5)
                # Add statistics
                stats_text = f"IMF{i + 1} Period Distribution\n" \
                             f"Median: {np.median(valid_periods):.1f} days\n" \
                             f"Mean: {np.mean(valid_periods):.1f} days\n" \
                             f"Std Dev: {np.std(valid_periods):.1f} days\n" \
                             f"Min: {np.min(valid_periods):.1f} days\n" \
                             f"Max: {np.max(valid_periods):.1f} days"
                ax.text(0.98, 0.95, stats_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                # Add reference periods
                for days, label in [(7, '1w'), (30, '1m'), (90, '3m'), (180, '6m'), (365, '1y')]:
                    ax.axvline(x=days, color='gray', linestyle='--', alpha=0.3)
            if log_scale:
                ax.set_xscale('log')
            ax.set_title(f'IMF {i + 1} Period Distribution', fontsize=12)
            ax.set_xlabel('Period (days)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{title.replace(" ", "_")}_imf_distributions.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
# Example usage
if __name__ == "__main__":
    # Get data
    signal = get_data('399300.SZ', '20200101', '20260101',
                      name='CSI300', fields=['close']).values.reshape(-1)
    dates = pd.date_range(start='20200101', periods=len(signal), freq='D')
    # Create analyzer instance
    analyzer = CEEMDANAnalyzer(signal, dates)
    # Define parameter bounds
    lb = [0.01, 10]  # [noise_std lower, ensemble_size lower]
    ub = [0.5, 200]  # [noise_std upper, ensemble_size upper]
    # Optimize and decompose
    imfs = analyzer.optimize_and_decompose(lb, ub, search_agents=10, max_iter=10)
    # Plot results
    analyzer.plot_convergence()
    analyzer.plot_ceemdan_results()
    analyzer.plot_period_analysis(title="CSI300 IMFs Period Analysis", log_scale=True)
    analyzer.plot_period_analysis(title="CSI300 IMFs Period Analysis", log_scale=False)