"""
ClonalEvolutionEngine: Tumor Clonal Evolution from Multi-Region Sequencing
- Cancer cell fraction (CCF) estimation from VAF + copy number
- Clonal hierarchy reconstruction (phylogenetic tree from CCF ordering)
- Subclone detection (Gaussian mixture model clustering of VAFs)
- Driver mutation timing (early vs late based on CCF)
- Evolutionary fitness estimation (subclone growth rate)
"""

import numpy as np
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Data Simulation ────────────────────────────────────────────────────────

N_TUMORS = 20
N_REGIONS = 5
N_MUTATIONS = 200

purity = np.random.uniform(0.6, 0.9, size=N_TUMORS)
cn_total = np.random.choice([1, 2, 2, 2, 3, 4], size=(N_TUMORS, N_MUTATIONS))
cn_alt = np.ones((N_TUMORS, N_MUTATIONS), dtype=int)

true_ccf_base = np.zeros((N_TUMORS, N_MUTATIONS))
true_ccf_base[:, :60] = np.random.uniform(0.85, 1.0, size=(N_TUMORS, 60))
true_ccf_base[:, 60:130] = np.random.uniform(0.3, 0.65, size=(N_TUMORS, 70))
true_ccf_base[:, 130:] = np.random.uniform(0.05, 0.3, size=(N_TUMORS, 70))

ccf_all = np.zeros((N_TUMORS, N_MUTATIONS, N_REGIONS))
for t in range(N_TUMORS):
    for r in range(N_REGIONS):
        noise = np.random.normal(0, 0.05, size=N_MUTATIONS)
        ccf_all[t, :, r] = np.clip(true_ccf_base[t] + noise, 0.01, 1.0)

vaf_all = np.zeros((N_TUMORS, N_MUTATIONS, N_REGIONS))
for t in range(N_TUMORS):
    p = purity[t]
    for r in range(N_REGIONS):
        vaf_all[t, :, r] = (ccf_all[t, :, r] * p * cn_alt[t]) / \
                            (cn_total[t] * p + 2 * (1 - p))
vaf_all = np.clip(vaf_all, 0.001, 0.999)

# ─── Algorithm 1: CCF Estimation ─────────────────────────────────────────────

def estimate_ccf(vaf, cn_total, cn_alt, purity):
    ccf = vaf * (cn_total / cn_alt) / purity
    return np.clip(ccf, 0.0, 1.0)

ccf_estimated = np.zeros_like(vaf_all)
for t in range(N_TUMORS):
    for r in range(N_REGIONS):
        ccf_estimated[t, :, r] = estimate_ccf(
            vaf_all[t, :, r], cn_total[t], cn_alt[t], purity[t])

# ─── Algorithm 2: Clonal vs Subclonal ────────────────────────────────────────

mean_ccf = ccf_estimated.mean(axis=2)
min_ccf = ccf_estimated.min(axis=2)

is_clonal = min_ccf > 0.8
clonal_counts = is_clonal.sum(axis=1)
subclonal_counts = N_MUTATIONS - clonal_counts

print(f"Mean clonal mutations per tumor: {clonal_counts.mean():.1f}")
print(f"Mean subclonal mutations per tumor: {subclonal_counts.mean():.1f}")

# ─── Algorithm 3: GMM Subclone Detection ─────────────────────────────────────

def gmm_em(data, n_components=3, n_iter=100, tol=1e-6):
    n = len(data)
    means = np.linspace(data.min() + 0.1, data.max() - 0.1, n_components)
    stds = np.full(n_components, 0.1)
    weights = np.ones(n_components) / n_components
    log_likelihood_prev = -np.inf
    responsibilities = np.zeros((n, n_components))

    for iteration in range(n_iter):
        for k in range(n_components):
            responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k])
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-300, row_sums)
        responsibilities /= row_sums

        Nk = responsibilities.sum(axis=0)
        Nk = np.where(Nk == 0, 1e-10, Nk)
        weights = Nk / n
        means = (responsibilities * data[:, None]).sum(axis=0) / Nk
        stds = np.sqrt((responsibilities * (data[:, None] - means)**2).sum(axis=0) / Nk)
        stds = np.clip(stds, 0.01, 0.5)

        ll = np.zeros(n)
        for k in range(n_components):
            ll += weights[k] * stats.norm.pdf(data, means[k], stds[k])
        log_likelihood = np.log(ll + 1e-300).sum()
        if abs(log_likelihood - log_likelihood_prev) < tol:
            break
        log_likelihood_prev = log_likelihood

    labels = responsibilities.argmax(axis=1)
    return labels, means, stds, weights

print("Running GMM subclone detection...")
example_tumor = 0
ccf_tumor0 = mean_ccf[example_tumor]
gmm_labels, gmm_means, gmm_stds, gmm_weights = gmm_em(ccf_tumor0, n_components=3)

subclone_counts_per_tumor = []
for t in range(N_TUMORS):
    labels, means, stds, weights = gmm_em(mean_ccf[t], n_components=3)
    n_sub = np.sum(weights > 0.05)
    subclone_counts_per_tumor.append(n_sub)
subclone_counts_per_tumor = np.array(subclone_counts_per_tumor)

# ─── Algorithm 4: Phylogenetic tree ──────────────────────────────────────────

def build_phylo_tree(ccf_matrix):
    mean_ccf_mut = ccf_matrix.mean(axis=1)
    order = np.argsort(-mean_ccf_mut)
    return order, mean_ccf_mut

order_t0, mean_ccf_t0 = build_phylo_tree(ccf_estimated[example_tumor])

ccf_for_linkage = ccf_estimated[example_tumor, :30, :]
dist_matrix = pdist(ccf_for_linkage, metric='euclidean')
Z = linkage(dist_matrix, method='ward')

# ─── Algorithm 5: Fitness estimation ─────────────────────────────────────────

def estimate_fitness(ccf_regions):
    fitness = ccf_regions[:, -1] / (ccf_regions[:, 0] + 1e-6)
    return fitness

fitness_scores = np.zeros((N_TUMORS, N_MUTATIONS))
for t in range(N_TUMORS):
    fitness_scores[t] = estimate_fitness(ccf_estimated[t])

mean_fitness = fitness_scores.mean(axis=1)

# ─── Driver mutation timing ───────────────────────────────────────────────────

n_early = np.sum(mean_ccf > 0.7, axis=1)
n_late = np.sum(mean_ccf < 0.4, axis=1)
n_intermediate = N_MUTATIONS - n_early - n_late

# ─── Dashboard ───────────────────────────────────────────────────────────────

print("Generating dashboard...")
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

COLORS = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a29bfe', '#fd79a8', '#74b9ff', '#55efc4']
TEXT_COLOR = 'white'
GRID_COLOR = '#333333'

def style_ax(ax, title):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

# Panel 1: CCF distribution across regions (violin)
ax1 = fig.add_subplot(gs[0, 0])
ccf_by_region = [ccf_estimated[:, :, r].flatten() for r in range(N_REGIONS)]
parts = ax1.violinplot(ccf_by_region, positions=range(N_REGIONS), showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(COLORS[0])
    pc.set_alpha(0.7)
parts['cmedians'].set_color(COLORS[1])
ax1.set_xlabel('Region', color=TEXT_COLOR, fontsize=8)
ax1.set_ylabel('CCF', color=TEXT_COLOR, fontsize=8)
ax1.set_xticks(range(N_REGIONS))
ax1.set_xticklabels([f'R{i+1}' for i in range(N_REGIONS)], color=TEXT_COLOR, fontsize=7)
style_ax(ax1, 'Panel 1: CCF Distribution by Region')

# Panel 2: Clonal vs subclonal proportions
ax2 = fig.add_subplot(gs[0, 1])
x2 = np.arange(N_TUMORS)
ax2.bar(x2, clonal_counts, color=COLORS[0], alpha=0.85, label='Clonal', edgecolor='#003322')
ax2.bar(x2, subclonal_counts, bottom=clonal_counts, color=COLORS[1], alpha=0.85,
        label='Subclonal', edgecolor='#330000')
ax2.set_xlabel('Tumor', color=TEXT_COLOR, fontsize=8)
ax2.set_ylabel('Mutation Count', color=TEXT_COLOR, fontsize=8)
ax2.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax2, 'Panel 2: Clonal vs Subclonal Mutations')

# Panel 3: Phylogenetic tree (dendrogram)
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#111111')
dendrogram(Z, ax=ax3, color_threshold=0.7*max(Z[:,2]),
           above_threshold_color=COLORS[4],
           link_color_func=lambda k: COLORS[4])
ax3.tick_params(colors=TEXT_COLOR, labelsize=6)
ax3.set_xlabel('Mutation', color=TEXT_COLOR, fontsize=8)
ax3.set_ylabel('Distance', color=TEXT_COLOR, fontsize=8)
for spine in ax3.spines.values():
    spine.set_edgecolor('#444444')
ax3.set_title('Panel 3: Phylogenetic Tree (Tumor 1)', fontsize=10,
              fontweight='bold', color=TEXT_COLOR, pad=8)
ax3.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

# Panel 4: GMM subclone clustering
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(range(N_MUTATIONS), ccf_tumor0, c=[COLORS[l % len(COLORS)] for l in gmm_labels],
            alpha=0.6, s=15)
for k in range(3):
    ax4.axhline(gmm_means[k], color=COLORS[k], linestyle='--', linewidth=1.5,
                label=f'Clone {k+1}: CCF={gmm_means[k]:.2f} (w={gmm_weights[k]:.2f})')
ax4.set_xlabel('Mutation Index', color=TEXT_COLOR, fontsize=8)
ax4.set_ylabel('Mean CCF', color=TEXT_COLOR, fontsize=8)
ax4.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax4, 'Panel 4: GMM Subclone Clustering (Tumor 1)')

# Panel 5: Driver mutation timing
ax5 = fig.add_subplot(gs[1, 1])
timing_data = np.array([n_early.mean(), n_intermediate.mean(), n_late.mean()])
timing_labels = ['Early\n(CCF>0.7)', 'Intermediate\n(0.4-0.7)', 'Late\n(CCF<0.4)']
bars5 = ax5.bar(timing_labels, timing_data, color=[COLORS[0], COLORS[3], COLORS[1]],
                alpha=0.85, edgecolor='#222222')
for bar, val in zip(bars5, timing_data):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=8)
ax5.set_ylabel('Mean Mutation Count', color=TEXT_COLOR, fontsize=8)
style_ax(ax5, 'Panel 5: Driver Mutation Timing')

# Panel 6: Fitness score distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(fitness_scores.flatten(), bins=40, color=COLORS[5], alpha=0.85, edgecolor='#330022')
ax6.axvline(1.0, color=COLORS[1], linestyle='--', linewidth=1.5, label='Neutral (1.0)')
ax6.axvline(np.mean(fitness_scores), color=COLORS[0], linestyle='-', linewidth=1.5,
            label=f'Mean={np.mean(fitness_scores):.2f}')
ax6.set_xlabel('Fitness Score (CCF ratio)', color=TEXT_COLOR, fontsize=8)
ax6.set_ylabel('Count', color=TEXT_COLOR, fontsize=8)
ax6.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax6, 'Panel 6: Evolutionary Fitness Distribution')

# Panel 7: CCF heatmap (mutations × regions, example tumor)
ax7 = fig.add_subplot(gs[2, 0])
ccf_heatmap = ccf_estimated[example_tumor, :50, :]
cmap_ccf = LinearSegmentedColormap.from_list('ccf', ['#111111', '#4ecdc4', '#ffe66d', '#ff6b6b'])
im7 = ax7.imshow(ccf_heatmap, aspect='auto', cmap=cmap_ccf, vmin=0, vmax=1)
cb7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='CCF')
cb7.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
ax7.set_xlabel('Region', color=TEXT_COLOR, fontsize=8)
ax7.set_ylabel('Mutation', color=TEXT_COLOR, fontsize=8)
ax7.set_xticks(range(N_REGIONS))
ax7.set_xticklabels([f'R{i+1}' for i in range(N_REGIONS)], color=TEXT_COLOR, fontsize=7)
style_ax(ax7, 'Panel 7: CCF Heatmap (Tumor 1, 50 muts)')

# Panel 8: Subclone count per tumor
ax8 = fig.add_subplot(gs[2, 1])
unique_sc, counts_sc = np.unique(subclone_counts_per_tumor, return_counts=True)
ax8.bar(unique_sc, counts_sc, color=COLORS[6], alpha=0.85, edgecolor='#002244')
ax8.set_xlabel('Number of Subclones', color=TEXT_COLOR, fontsize=8)
ax8.set_ylabel('Number of Tumors', color=TEXT_COLOR, fontsize=8)
style_ax(ax8, 'Panel 8: Subclone Count per Tumor')

# Panel 9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#111111')
ax9.axis('off')
for spine in ax9.spines.values():
    spine.set_edgecolor('#444444')

summary_lines = [
    "CLONAL EVOLUTION ENGINE SUMMARY",
    "─" * 34,
    f"Tumors analyzed:       {N_TUMORS}",
    f"Regions per tumor:     {N_REGIONS}",
    f"Mutations per tumor:   {N_MUTATIONS}",
    f"Mean purity:           {purity.mean():.3f}",
    f"Mean clonal muts:      {clonal_counts.mean():.1f}",
    f"Mean subclonal muts:   {subclonal_counts.mean():.1f}",
    f"Clonal fraction:       {clonal_counts.mean()/N_MUTATIONS:.3f}",
    f"Mean subclone count:   {subclone_counts_per_tumor.mean():.2f}",
    f"GMM clone 1 CCF:       {sorted(gmm_means)[-1]:.3f}",
    f"GMM clone 2 CCF:       {sorted(gmm_means)[-2]:.3f}",
    f"GMM clone 3 CCF:       {sorted(gmm_means)[-3]:.3f}",
    f"Mean fitness score:    {np.mean(fitness_scores):.3f}",
    f"Early driver muts:     {n_early.mean():.1f}",
    f"Late driver muts:      {n_late.mean():.1f}",
]
ax9.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax9.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
ax9.set_title('Panel 9: Summary', fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=8)

fig.suptitle('ClonalEvolutionEngine: Tumor Clonal Evolution Dashboard',
             fontsize=14, fontweight='bold', color=TEXT_COLOR, y=0.98)

plt.savefig('/workspace/subagents/7c45dd59/clonal_evolution_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a', edgecolor='none')
plt.close()
print("Dashboard saved: /workspace/subagents/7c45dd59/clonal_evolution_dashboard.png")

# ─── Structured Summary ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("CLONAL EVOLUTION ENGINE — STRUCTURED SUMMARY")
print("="*60)
print(f"Tumors analyzed:                 {N_TUMORS}")
print(f"Regions per tumor:               {N_REGIONS}")
print(f"Mutations per tumor:             {N_MUTATIONS}")
print(f"Mean tumor purity:               {purity.mean():.3f} ± {purity.std():.3f}")
print(f"Mean clonal mutations:           {clonal_counts.mean():.1f} ± {clonal_counts.std():.1f}")
print(f"Mean subclonal mutations:        {subclonal_counts.mean():.1f} ± {subclonal_counts.std():.1f}")
print(f"Mean clonal fraction:            {clonal_counts.mean()/N_MUTATIONS:.3f}")
print(f"Mean subclone count per tumor:   {subclone_counts_per_tumor.mean():.2f}")
print(f"GMM clone CCF centers (T1):      {[round(x,3) for x in sorted(gmm_means, reverse=True)]}")
print(f"GMM clone weights (T1):          {[round(x,3) for x in gmm_weights[np.argsort(-gmm_means)].tolist()]}")
print(f"Mean evolutionary fitness:       {np.mean(fitness_scores):.3f} ± {np.std(fitness_scores):.3f}")
print(f"Mean early driver mutations:     {n_early.mean():.1f}")
print(f"Mean late driver mutations:      {n_late.mean():.1f}")
print(f"Mean CCF (all tumors/regions):   {ccf_estimated.mean():.3f}")
print("="*60)
