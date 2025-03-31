from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import scipy

# Parameters
alpha = 0.1
Ks = [20, 50, 100, 200]
null_props = [0.5, 0.7, 0.9]
signal_strengths = [2, 3, 4]
n_trials = 1000
n_cores = 12


def harmonic_num(K):
    return np.sum(1 / np.arange(1, K + 1))

def BY_calibrator(pvalues, alpha, K):
    ell_K = harmonic_num(K)
    evalues = np.zeros(pvalues.shape)
    incr = alpha / (K * ell_K)
    evalues[pvalues <= ell_K / alpha] = K / (alpha * np.maximum(np.ceil(pvalues / incr), 1))
    return evalues

def worst_BY_pvalues(K: int, alpha: float, pi_0: float, seed: Optional[float] = None):
    """P-value joint distribution that is tight with the BY FDR bound (under
    arbitrary dependence)

    :param K: number of hypotheses
    :param alpha: FDR of BY applied to p-values
    :param pi_0: proportion of null hypotheses (take ceiling if
        np.ceil(pi_0 * non- integer))
    :param trials: number of trials
    :param seed: rng seed
    :return: trials x K array of p-values
    """
    rng = np.random.default_rng(seed)
    m_0 = int(np.ceil(pi_0 * K))

    thresh = alpha / harmonic_num(K)
    coef = thresh / K
    null_p = (m_0 * coef) / np.arange(1, m_0 + 1)
    alt_p = np.full((K - m_0, ), coef)
    rem_p = 1 - (np.sum(null_p) + np.sum(alt_p))
    p = np.concatenate([null_p, alt_p, np.array([rem_p])])
    N = rng.choice(np.arange(K + 1) + 1, p=p)
    u = rng.uniform(size=(2,))
    P = np.zeros(shape=(K))

    null_indices = np.arange(m_0)
    alt_indices = np.arange(m_0, K)

    N = N
    U_Kp1 = thresh + (u[0] * (1 - thresh))
    if N <= m_0:
        N_indices = rng.choice(null_indices, size=N, replace=False)
        N_mask = np.zeros(K).astype(bool)
        N_mask[N_indices] = True
        P[N_mask] = coef * (N + u[1] - 1)
        P[~N_mask] = U_Kp1
    elif N < K + 1:
        N_indices = rng.choice(alt_indices,
                               size=N - m_0,
                               replace=False)
        N_mask = np.zeros(K).astype(bool)
        N_mask[N_indices] = True
        U_N = coef * (N + u[1] - 1)
        P[:m_0] = U_N
        P[N_mask] = U_N
        P[(~N_mask) & (np.arange(K) > m_0)] = U_Kp1
    else:
        P[:m_0] = U_Kp1
        P[m_0:] = 1.

    # alternates = np.concatenate([np.zeros(m_0), np.ones(K - m_0)])
    return P, set(range(m_0))


# def posthoc_ebh_evalues(evalues):
#     eval_indices = np.argsort(evalues)
#     sorted_evalues = evalues[eval_indices]
#     min_alphas = np.zeros(evalues.shape)
#     for i, idx in enumerate(eval_indices):
#         # either the min_alpha was the previous smaller one, or you're the smallest one in the discover set
#         min_alphas[i] = min((K / (K - i)) / evalues[idx], min_alphas[i - 1] if i >= 1 else 1)
#
#     for k in range(K, 0, -1):
#         for r in range(1, k):
#             for d_m in range(0, K - k + 1):
#                 m = r + d_m
#                 slac
#
#
#
#
#
#
#     out_evalues = np.zeros(pvalues.shape)
#     incr = alpha / K
#     evalues[evalues >= 1 / alpha] = K / (alpha * np.maximum(np.ceil((1 / evalues) * incr), 1))

cov_cache = {}
def simulate_e_values(K, null_prop, signal_strength, mode='simple', alpha=None):
    # lambda_e = np.sqrt(np.log(2 * K / alpha))
    # lambda_e = np.sqrt(np.log(2 * K / alpha))
    lambda_e = signal_strength
    n_null = int(K * null_prop)
    n_alt = K - n_null
    Z = np.zeros(K)

    if mode == 'simple':
        Z[:n_null] = np.random.normal(0, 1, n_null)
        Z[n_null:] = np.random.normal(signal_strength, 1, n_alt)
        E = np.exp(lambda_e * Z - (lambda_e ** 2) / 2)
    elif mode == 'simple-BY':
        assert alpha is not None
        # Create a fixed covariance matrix with both positive and negative correlations
        # that is guaranteed to be positive semi-definite

        # Define a fixed correlation pattern matrix
        # This Toeplitz-like structure guarantees a valid PSD matrix
        if K not in cov_cache:
            cov_matrix = np.eye(K)  # Start with identity matrix
            for i in range(K):
                for j in range(i+1, K):
                    dist = j - i
                    # Alternating positive and negative correlations that decay with distance
                    if dist % 2 == 0:
                        cov_matrix[i, j] = 0.2 * np.exp(-dist/10)
                    else:
                        cov_matrix[i, j] = -0.2 * np.exp(-dist/10)
                    cov_matrix[j, i] = cov_matrix[i, j]  # Ensure symmetry
            cov_cache[K] = cov_matrix
        else:
            cov_matrix = cov_cache[K]


        # Generate Z from multivariate normal with the constructed covariance
        Z = np.random.multivariate_normal(
            mean=np.concatenate([np.zeros(n_null), np.full(K-n_null, signal_strength)]),
            cov=cov_matrix
        )
        pvalues = 1 - scipy.stats.norm.cdf(Z)
        E = BY_calibrator(pvalues, alpha, K)
    elif mode == "worst-case-BY":
        assert alpha is not None
        pvalues = worst_BY_pvalues(K, alpha, pi_0 = null_prop)
        E = BY_calibrator(pvalues, alpha, K)
    elif mode == "simple-posthoc-eBH":
        assert alpha is not None
        pvalues = worst_BY_pvalues(K, alpha, pi_0 = null_prop)
        E = BY_calibrator(pvalues, alpha, K)
    else:
        # version == 'compound':
        variances = np.random.uniform(0.5, 3, K)
        Z[:n_null] = np.random.normal(0, variances[:n_null])
        Z[n_null:] = np.random.normal(2 * variances[n_null:] * (np.random.normal(np.sqrt(signal_strength), 1) ** 2), variances[n_null:])
        E = K * (Z ** 2) / np.sum(variances)
    return E, set(range(n_null))




def compute_cebh_discovery_set(E, alpha, mode='simple'):
    K = len(E)
    R_ebh = compute_ebh(E, alpha)
    ebh_k = len(R_ebh)

    sorted_idx = np.argsort(E)[::-1] # sorted largest to smallest
    E_sorted = E[sorted_idx]

    E_prefix_sum = np.cumsum(E_sorted)
    # print(E_prefix_sum)
    E_suffix_sum = np.cumsum(E_sorted[::-1])


    res_dict = {}
    def get_r_of_k(r, k):
        if (r, k) not in res_dict:
            res_dict[(r, k)] = np.sum(E_sorted[(k - r):k])
        return res_dict[(r, k)]

    def get_evalue_fdp(k, r, m):
        """k is number of discoveries, r is number of false discoveries, m is number of nulls"""
        rejected_sum = get_r_of_k(r, k)
        nonreject_sum = E_suffix_sum[m - r - 1] if m > r else 0
        evalue = (rejected_sum + nonreject_sum) / m
        fdp = r / k
        return evalue, fdp
    def check_e_safety(k, r, m):
        evalue, fdp = get_evalue_fdp(k, r, m)
        return fdp <= alpha * evalue


    for k in range(K, ebh_k, -1):
        valid = True
        for r in range(1, k + 1): # of false discoveries
            for m in range(r, r + K + 1 - k): # total null count
                if not check_e_safety(k, r, m):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return set(sorted_idx[:k])
    return R_ebh


def compute_ebh(E, alpha, min_adaptive=False):
    K = len(E)
    sorted_idx = np.argsort(E)[::-1]
    if min_adaptive:
        if np.mean(E) >= (1 / alpha):
            alpha = alpha * (K / (K - 1))
        else:
            return set()
    for i in range(K, 0, -1):
        threshold = K / (alpha * i)
        if E[sorted_idx[i - 1]] >= threshold:
            return set(sorted_idx[:i])
    return set()


def compute_estorey(E, alpha, lambda_thresh=0.9):
    K = len(E)
    num = 1 + len([e for e in E if e < 1 / lambda_thresh])
    pi0_hat = num / ((1 - lambda_thresh) * K)
    E /= pi0_hat
    sorted_idx = np.argsort(E)[::-1]
    for i in range(K, 0, -1):
        threshold =  K / (alpha * i)
        if E[sorted_idx[i - 1]] >= threshold:
            return set(sorted_idx[:i])
    return set()


def single_simulation(args):
    K, null_prop, signal_strength, mode = args
    fdr_bnp, tdp_bnp = [], []
    fdr_ebh, tdp_ebh = [], []
    fdr_ebh_adapt, tdp_ebh_adapt = [], []
    fdr_estorey, tdp_estorey = [], []

    for _ in range(n_trials):
        E, true_nulls = simulate_e_values(K, null_prop, signal_strength, mode=mode, alpha=alpha)
        R_bnp = compute_cebh_discovery_set(E, alpha, mode=mode)
        R_ebh = compute_ebh(E, alpha, min_adaptive=False)
        R_ebh_adapt = compute_ebh(E, alpha, min_adaptive=True)
        #R_estorey = compute_estorey(E, alpha)

        fdp_bnp = len(R_bnp & true_nulls) / max(len(R_bnp), 1)
        tdp_bnp.append((len(R_bnp) - len(R_bnp & true_nulls)) / (K - len(true_nulls)))
        fdr_bnp.append(fdp_bnp)

        fdp_ebh = len(R_ebh & true_nulls) / max(len(R_ebh), 1)
        tdp_ebh.append((len(R_ebh) - len(R_ebh & true_nulls)) / (K - len(true_nulls)))
        fdr_ebh.append(fdp_ebh)

        fdp_ebh_adapt = len(R_ebh_adapt & true_nulls) / max(len(R_ebh_adapt), 1)
        tdp_ebh_adapt.append((len(R_ebh_adapt) - len(R_ebh_adapt & true_nulls)) / (K - len(true_nulls)))
        fdr_ebh_adapt.append(fdp_ebh_adapt)

        #fdp_estorey = len(R_estorey & true_nulls) / max(len(R_estorey), 1)
        #tdp_estorey.append((len(R_estorey) - len(R_estorey & true_nulls)) / (K - len(true_nulls)))
        #fdr_estorey.append(fdp_estorey)

    return {
        'K': K, 'null_prop': null_prop, 'signal': signal_strength,
        'FDR_BNP': np.mean(fdr_bnp), 'TDP_BNP': np.mean(tdp_bnp),
        'FDR_eBH': np.mean(fdr_ebh), 'TDP_eBH': np.mean(tdp_ebh),
        'FDR_eBH_adapt': np.mean(fdr_ebh_adapt), 'TDP_eBH_adapt': np.mean(tdp_ebh_adapt),
#        'FDR_eStorey': np.mean(fdr_estorey), 'TDP_eStorey': np.mean(tdp_estorey),
    }


def run_simulation(mode):
    param_grid = [(K, null_prop, signal_strength, mode)
                  for K in Ks
                  for null_prop in null_props
                  for signal_strength in signal_strengths]

    with Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(single_simulation, param_grid), total=len(param_grid), desc="Running simulations"))
    return results

def plot_results_BY(results, save_path=None):
    df = pd.DataFrame(results)
    sns.set(style="whitegrid")
    figs = []

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    light_blue = '#1f77b4'
      # usually the light blue
    for null_prop in null_props:
        for signal in signal_strengths:
            subset = df[(df['null_prop'] == null_prop) & (df['signal'] == signal)]

            fig1, ax1 = plt.subplots(figsize=(4.5, 3.3))
            print(f'signal: {signal}, null_prop: {null_prop}')
            print(subset)
            ax1.plot(subset['K'], subset['TDP_BNP'], label='$\\overline{\\mathrm{BY}}$\n(ours)', marker='o', color='orange')
            ax1.plot(subset['K'], subset['TDP_eBH'], label='BY', marker='s', color=light_blue)
            ax1.plot(subset['K'], subset['TDP_eBH_adapt'], label='BYm', marker="^", color='pink')
            ax1.axhline(y=alpha, color='red', linestyle='--', linewidth=1, label=f"$\\alpha={alpha:.1f}$")
            ax1.plot(subset['K'], subset['FDR_BNP'],linestyle='--', marker='o', color='orange')
            ax1.plot(subset['K'], subset['FDR_eBH'],linestyle='--', marker='s', color=light_blue)
            ax1.plot(subset['K'], subset['FDR_eBH_adapt'], linestyle='--', marker='^', color='pink')
            ax1.text(ax1.get_xlim()[1], alpha, 'TDP $\\uparrow$', ha='right', va='bottom')
            ax1.text(ax1.get_xlim()[1], 0.8 * alpha, 'FDR $\\downarrow$', ha='right', va='top')


            ax1.set_title(f'$\\pi_0$={null_prop}, $\\mu$={signal}')
            ax1.set_xlabel('$K$')
            ax1.set_ylabel('FDR / TDP')
            ax1.set_ylim((0, 1))
            # Tight layout with space on the right for the legend
            fig1.tight_layout(rect=[0, 0, 0.75, 1])  # leave space on the right

            # Add the legend outside the plot
            fig1.legend(loc='center left', bbox_to_anchor=(0.72, 0.5))

            if save_path:
                fig1.savefig(f"{save_path}/null{null_prop}_signal{signal}.pdf", dpi=300)

    return figs
def plot_results(results, save_path=None):
    df = pd.DataFrame(results)
    sns.set(style="whitegrid")
    figs = []

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    light_blue = '#1f77b4'
      # usually the light blue
    for null_prop in null_props:
        for signal in signal_strengths:
            subset = df[(df['null_prop'] == null_prop) & (df['signal'] == signal)]

            fig1, ax1 = plt.subplots(figsize=(4.5, 3.3))
            print(f'signal: {signal}, null_prop: {null_prop}')
            print(subset)
            ax1.plot(subset['K'], subset['TDP_BNP'], label='$\\overline{\\mathrm{eBH}}$\n(ours)', marker='o', color='orange')
            ax1.plot(subset['K'], subset['TDP_eBH'], label='eBH', marker='s', color=light_blue)
            ax1.plot(subset['K'], subset['TDP_eBH_adapt'], label='eBHm', marker="^", color='pink')
            ax1.axhline(y=alpha, color='red', linestyle='--', linewidth=1, label=f"$\\alpha={alpha:.1f}$")
            ax1.plot(subset['K'], subset['FDR_BNP'],linestyle='--', marker='o', color='orange')
            ax1.plot(subset['K'], subset['FDR_eBH'],linestyle='--', marker='s', color=light_blue)
            ax1.plot(subset['K'], subset['FDR_eBH_adapt'], linestyle='--', marker='^', color='pink')
            ax1.text(ax1.get_xlim()[1], alpha, 'TDP $\\uparrow$', ha='right', va='bottom')
            ax1.text(ax1.get_xlim()[1], 0.8 * alpha, 'FDR $\\downarrow$', ha='right', va='top')


            ax1.set_title(f'$\\pi_0$={null_prop}, $\\mu$={signal}')
            ax1.set_xlabel('$K$')
            ax1.set_ylabel('FDR / TDP')
            ax1.set_ylim((0, 1))
            # Tight layout with space on the right for the legend
            fig1.tight_layout(rect=[0, 0, 0.75, 1])  # leave space on the right

            # Add the legend outside the plot
            fig1.legend(loc='center left', bbox_to_anchor=(0.72, 0.5))

            if save_path:
                fig1.savefig(f"{save_path}/null{null_prop}_signal{signal}.pdf", dpi=300)

    return figs

modes = [('simple', plot_results), ('simple-BY', plot_results_BY)]
for mode, plot_fn in modes:
    # Run simulations and plots
    save_dir = f"figures/{mode}"
    csv_path = f"{save_dir}/simulation_results.csv"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(csv_path):
        print("CSV found, loading cached results...")
        df_cached = pd.read_csv(csv_path)
        plot_fn(df_cached.to_dict(orient='records'), save_path=save_dir)
    else:
        results = run_simulation(mode=mode)
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        plot_fn(results, save_path=save_dir)
