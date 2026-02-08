import numpy as np
import os
import csv
from scipy.signal import convolve2d

# =====================
# Paths
# =====================
BASE_DIR = os.path.expanduser("~/Desktop/brainres_sim_bestgen")

# =====================
# Parameters
# =====================
GRID = 20
POP = 80
GEN = 80
LINEAGES = 30
ELITE = 2

TRIALS_MAIN = 32
TRIALS_REFINE = 64

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
# Environments
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
# Fitness (latency proxy)
# =====================
def fitness(ind, envs):
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
# Evolution (best-of-generation)
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

        scores = np.array([fitness(ind, envs) for ind in pop])

        # --- best-of-generation ---
        best = pop[np.argmax(scores)]
        history.append(abs(best[0] - best[1]))

        elite_idx = np.argsort(scores)[-ELITE:]
        survivors = np.argsort(scores)[-(POP // 2):]

        refined_scores = []
        for idx in survivors:
            envs_ref = [env_func(rng) for _ in range(TRIALS_REFINE)]
            refined_scores.append(fitness(pop[idx], envs_ref))
        refined_scores = np.array(refined_scores)

        selected = survivors[np.argsort(refined_scores)]
        next_pop = pop[selected[-(POP - ELITE):]]

        offspring = []
        for i in range(0, len(next_pop), 2):
            p1 = next_pop[i]
            p2 = next_pop[(i + 1) % len(next_pop)]
            child = (p1 + p2) / 2
            child += rng.normal(0, sd, 3)
            for j in [0, 1]:
                if child[j] < 0.1:
                    child[j] = 0.1 + (0.1 - child[j])
            offspring.append(child)

        pop = np.vstack([pop[elite_idx], np.array(offspring)[:POP - ELITE]])

    return np.array(history)

# =====================
# Run simulations
# =====================
all_spatial = []
all_planar = []

for i in range(LINEAGES):
    seed = 3000 + i
    all_spatial.append(evolve(seed, spatial_env))
    all_planar.append(evolve(seed, planar_env))

all_spatial = np.array(all_spatial)
all_planar = np.array(all_planar)

mean_spatial = np.mean(all_spatial, axis=0)
mean_planar = np.mean(all_planar, axis=0)

# =====================
# Save CSV
# =====================
out_path = os.path.join(BASE_DIR, "generation_stats_best.csv")

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["generation", "spatial_best", "planar_best"])
    for g in range(GEN):
        writer.writerow([g, mean_spatial[g], mean_planar[g]])

print("generation_stats_best.csv saved to", BASE_DIR)
