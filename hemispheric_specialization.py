import numpy as np
import os
import csv
from scipy.signal import convolve2d
from scipy.stats import wilcoxon

# =====================
# Save directory
# =====================
BASE_DIR = os.path.expanduser("~/Desktop/brainres_sim")
os.makedirs(BASE_DIR, exist_ok=True)

# =====================
# Global parameters
# =====================
GRID = 20
PIXELS = GRID * GRID
POP = 80
GEN = 80
LINEAGES = 30
ELITE = 2

TRIALS_MAIN = 32
TRIALS_REFINE = 64
TRIALS_TEST = 128

INIT_SD = 0.08
SD_DECAY = 0.97
EPS = 1e-6

# =====================
# Gaussian kernel
# =====================
def gaussian_kernel(sigma, size=7):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

# =====================
# Environment generation
# =====================
def spatial_env(rng):
    img = rng.normal(0, 1, (GRID, GRID))
    img = convolve2d(img, gaussian_kernel(1.0), mode="same")
    x, y = rng.integers(0, GRID-3, 2)
    img[x:x+3, y:y+3] += rng.uniform(1.5, 2.0)
    return img

def planar_env(rng):
    img = rng.normal(0, 1, (GRID, GRID))
    img -= img.mean()
    img /= img.std() + EPS
    return img

# =====================
# Evaluation
# =====================
def latency_proxy(ind, envs):
    aL, aR, lam = ind
    kL = gaussian_kernel(aL)
    kR = gaussian_kernel(aR)
    vals = []
    for I in envs:
        IL = convolve2d(I, kL, mode="same")
        IR = convolve2d(I, kR, mode="same")
        ML = np.mean(np.abs(I - IL))
        MR = np.mean(np.abs(I - IR))
        D = (MR - ML) + lam * (abs(MR) - abs(ML))
        vals.append(1.0 / (abs(D) + EPS))
    return np.mean(vals)

# =====================
# Evolution
# =====================
def evolve(seed, env_func):
    rng = np.random.default_rng(seed)
    pop = np.column_stack([
        rng.uniform(1.4, 1.6, POP),
        rng.uniform(1.4, 1.6, POP),
        rng.uniform(0.2, 0.4, POP)
    ])

    history = []

    for g in range(GEN):
        sd = INIT_SD * (SD_DECAY ** g)
        envs = [env_func(rng) for _ in range(TRIALS_MAIN)]
        scores = np.array([-latency_proxy(ind, envs) for ind in pop])

        elite_idx = np.argsort(scores)[-ELITE:]
        survivors = np.argsort(scores)[-(POP//2):]

        refined = []
        for idx in survivors:
            envs_ref = [env_func(rng) for _ in range(TRIALS_REFINE)]
            refined.append(-latency_proxy(pop[idx], envs_ref))
        refined = np.array(refined)

        selected = survivors[np.argsort(refined)]
        next_pop = pop[selected[-(POP-ELITE):]]

        offspring = []
        for i in range(0, len(next_pop), 2):
            p1, p2 = next_pop[i], next_pop[(i+1) % len(next_pop)]
            child = (p1 + p2) / 2
            child += rng.normal(0, sd, 3)
            for j in [0, 1]:
                if child[j] < 0.1:
                    child[j] = 0.1 + (0.1 - child[j])
            offspring.append(child)

        pop = np.vstack([pop[elite_idx], np.array(offspring)[:POP-ELITE]])
        history.append(np.mean(np.abs(pop[:,0] - pop[:,1])))

    test_envs = [env_func(rng) for _ in range(TRIALS_TEST)]
    best = pop[np.argmax([-latency_proxy(ind, test_envs) for ind in pop])]
    return best, history

# =====================
# Run simulation
# =====================
final_rows = []
table_rows = []

for i in range(LINEAGES):
    seed = 1000 + i
    best_s, hist_s = evolve(seed, spatial_env)
    best_p, hist_p = evolve(seed, planar_env)

    d_s = abs(best_s[0] - best_s[1])
    d_p = abs(best_p[0] - best_p[1])

    final_rows.append([i, "spatial", *best_s, d_s])
    final_rows.append([i, "planar", *best_p, d_p])

    table_rows.append([i, d_s, d_p])

# =====================
# Save CSV
# =====================
with open(os.path.join(BASE_DIR, "final_individuals.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lineage","condition","a_L","a_R","lambda","abs_delta"])
    writer.writerows(final_rows)

with open(os.path.join(BASE_DIR, "table1_summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lineage","delta_spatial","delta_planar"])
    writer.writerows(table_rows)

# stats
dS = np.array([r[1] for r in table_rows])
dP = np.array([r[2] for r in table_rows])
stat, p = wilcoxon(dS, dP)

with open(os.path.join(BASE_DIR, "stats.txt"), "w") as f:
    f.write(f"Wilcoxon p = {p}\n")
    f.write(f"Median spatial = {np.median(dS)}\n")
    f.write(f"Median planar = {np.median(dP)}\n")

print("Simulation complete. Results saved to Desktop/brainres_sim")

