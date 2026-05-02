# ALGORITMOS-DE-ENJAMBRE-EN-EL-APRENDIZAJE-M-QUINA
# Swarm Intelligence Applied to Machine Learning
### Dataset: Mushroom Classification (UCI Machine Learning Repository)

This project applies four **bio-inspired swarm intelligence algorithms** to different machine learning tasks using the UCI Mushroom dataset (8,124 samples, 22 categorical features, binary classification: edible vs. poisonous).

---

## Table of Contents

1. [Feature Selection with ABC](#1-feature-selection-with-abc)
2. [Hyperparameter Tuning with PSO](#2-hyperparameter-tuning-with-pso)
3. [Neural Network Training without Backpropagation (PSO)](#3-neural-network-training-without-backpropagation)
4. [Swarm Clustering (PSO, ABC, GWO, PSO-KModes)](#4-swarm-clustering)

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 8,124 |
| Features | 22 categorical |
| Classes | Edible (e): 4,208 — Poisonous (p): 3,916 |
| Missing values | Replaced with column mode |
| Encoding | Label Encoding |

---

## 1. Feature Selection with ABC

**File:** `abc_feature_selection.ipynb`  
**Algorithm:** Artificial Bee Colony (ABC)  
**Classifier:** Decision Tree (`max_depth=10`)

### How it works

Each bee solution is a **binary vector** of length 22 where `1` = feature selected and `0` = feature discarded. The fitness function balances classification accuracy against the number of selected features:

```
Fitness = CV_Accuracy - alpha * (n_selected / n_total)
```

The colony cycles through three phases:
- **Employed bees** — explore the neighborhood of their food source via bit-flip mutation.
- **Onlooker bees** — select food sources via roulette-wheel selection and refine them.
- **Scout bees** — abandon stagnant sources (no improvement after `limit` trials) and reinitialize randomly.

### Parameters

| Parameter | Value |
|---|---|
| `n_bees` | 20 |
| `max_iter` | 50 |
| `limit` | 8 |
| `alpha` | 0.05 |
| `cv` | 5-fold stratified |

### Results

| Metric | All 22 Features | ABC (4 Features) |
|---|---|---|
| Test Accuracy | 100.00% | 99.82% |
| Features used | 22 | **4** |
| Reduction | — | **81.8%** |

**Selected features:** `bruises`, `odor`, `stalk-surface-below-ring`, `habitat`

The algorithm reduced the feature space by 81.8% with only a 0.185% drop in accuracy, demonstrating that mushroom toxicity can be reliably predicted from just 4 out of 22 attributes.

---

## 2. Hyperparameter Tuning with PSO

**File:** `pso_hyperparameter_tuning.ipynb`  
**Algorithm:** Particle Swarm Optimization (PSO)  
**Model:** Random Forest Classifier

### How it works

Each particle is a **5-dimensional continuous vector** encoding Random Forest hyperparameters. PSO minimizes the cross-validation error (1 − accuracy) by updating particle velocities and positions according to:

```
v(t+1) = w·v(t) + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x(t+1) = x(t) + v(t+1)
```

Inertia `w` decays linearly from 0.9 to 0.4 across iterations to shift from global exploration to local exploitation.

### Search Space

| Dimension | Hyperparameter | Range |
|---|---|---|
| 0 | `n_estimators` | [10, 300] |
| 1 | `max_depth` | [2, 30] |
| 2 | `min_samples_split` | [2, 20] |
| 3 | `min_samples_leaf` | [1, 10] |
| 4 | `max_features` | [0.1, 1.0] |

### Parameters

| Parameter | Value |
|---|---|
| `n_particles` | 20 |
| `n_iter` | 40 |
| `w` (initial) | 0.9 → 0.4 (linear decay) |
| `c1`, `c2` | 1.5 |
| `cv` | 3-fold stratified |

### Results

| Model | 5-Fold CV Accuracy | Test Accuracy |
|---|---|---|
| Default Random Forest | 100.00% | 100.00% |
| PSO-tuned Random Forest | 100.00% | 100.00% |

**Best hyperparameters found:**

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 209 |
| `max_depth` | 27 |
| `min_samples_split` | 7 |
| `min_samples_leaf` | 7 |
| `max_features` | 0.944 |

The PSO converged in iteration 1, achieving perfect accuracy. The mushroom dataset is highly separable, meaning Random Forest reaches a ceiling regardless of hyperparameter configuration. PSO's value here is the principled, automatic search that would matter significantly on harder datasets.

---

## 3. Neural Network Training without Backpropagation

**File:** `pso_neural_network.ipynb`  
**Algorithm:** Particle Swarm Optimization (PSO)  
**Model:** Feedforward Neural Network (trained entirely by PSO — no gradient descent)

### Architecture

```
Input (22) → Hidden 1 (16, ReLU) → Hidden 2 (8, ReLU) → Output (1, Sigmoid)
```

Total trainable parameters: **513**

### How it works

Each particle encodes the **entire weight and bias vector** of the network (513 values). PSO searches the weight space directly, optimizing a composite fitness:

```
Fitness = 0.7 * Binary Cross-Entropy + 0.3 * Error Rate
```

No gradients are computed at any point. The update rule uses the standard PSO velocity equation with:
- Inertia decaying from 0.9 → 0.4
- Velocity clipped to `[-0.5, 0.5]`
- Weights clipped to `[-2.0, 2.0]`

### Parameters

| Parameter | Value |
|---|---|
| `n_particles` | 50 |
| `max_iter` | 200 |
| `w` | 0.9 → 0.4 |
| `c1`, `c2` | 2.0 |
| `v_max` | 0.5 |
| Convergence threshold | 0.01 |

### Training Progress

| Iteration | Fitness | Train Acc | Test Acc |
|---|---|---|---|
| 1 | 0.453 | 78.87% | 80.31% |
| 40 | 0.216 | 91.21% | 91.94% |
| 100 | 0.115 | 96.38% | 96.49% |
| 160 | 0.069 | 97.78% | 97.60% |
| 200 | 0.046 | 97.88% | 98.09% |

### Final Results

| Metric | Value |
|---|---|
| Training time | ~30 seconds |
| Final fitness | 0.0465 |
| Train accuracy | **97.88%** |
| Test accuracy | **98.09%** |
| Precision (poisonous) | 0.99 |
| Recall (poisonous) | 0.97 |

This demonstrates that a neural network can be trained to near-99% accuracy using only swarm intelligence — no backpropagation required.

---

## 4. Swarm Clustering

**File:** `swarm_clustering.ipynb`  
**Algorithms:** PSO, ABC, GWO, PSO-KModes  
**Task:** Unsupervised clustering (recover the edible/poisonous structure without labels)

Data was preprocessed using frequency encoding, standardization, and PCA (5 components, 63.1% variance explained). A stratified sample of 1,500 points was used for efficiency.

---

### 4.1 PSO Clustering (k = 2)

Each particle represents **k concatenated centroids** (k × d continuous values). Fitness = total inertia (sum of squared distances from each point to its nearest centroid). Centroids are initialized with k-means++ for better starting positions.

**Parameters:** 40 particles, 200 iterations, w = 0.5, c1 = c2 = 2.0

| Metric | Value |
|---|---|
| Final Inertia | 15,666.4 |
| Adjusted Rand Index | **0.5854** |
| Silhouette Score | 0.2706 |
| Cluster 0 | 912 pts — 759 edible, 153 poisonous |
| Cluster 1 | 588 pts — 23 edible, 565 poisonous |

---

### 4.2 ABC Clustering (automatic k selection)

The ABC variant searches for both the **optimal k** and the best centroids simultaneously. Each food source encodes a variable number of centroids (k ∈ {2, 3, 4, 5}). Fitness = Davies-Bouldin index (lower = better-separated clusters). Scout bees reinitialize stagnant sources, enabling automatic k discovery.

**Parameters:** 20 bees, 120 cycles, limit = 6, k_min = 2, k_max = 5

| Metric | Value |
|---|---|
| Optimal k found | **5** |
| Davies-Bouldin Score | 0.7129 |
| Adjusted Rand Index | 0.2408 |
| Silhouette Score | **0.5203** |

The higher Silhouette score (0.52 vs 0.27 for PSO) shows ABC found geometrically cleaner clusters, though with lower ARI because 5 clusters don't map directly to the 2 true classes.

---

### 4.3 GWO Clustering (k = 2)

Grey Wolf Optimizer mimics the **alpha–beta–delta–omega hierarchy** of wolf packs. Each wolf (particle) represents k centroids. Movement is guided by the three best solutions:

```
X(t+1) = (X1 + X2 + X3) / 3
```

where X1, X2, X3 are positions steered by alpha, beta, and delta respectively. A 20% random reinitialization of omega wolves maintains diversity.

**Parameters:** 30 wolves, 200 iterations, `a` decays from 2 → 0

| Metric | Value |
|---|---|
| Final Inertia | 15,431.8 |
| Adjusted Rand Index | 0.0939 |
| Silhouette Score | 0.3465 |

GWO achieved the lowest inertia of the three k=2 algorithms but had the worst ARI, indicating it found compact clusters that don't align well with the true class boundaries.

---

### 4.4 PSO-KModes (k = 2, categorical data)

PSO-KModes operates directly on the **original categorical features** (no PCA/scaling) using **Hamming distance** instead of Euclidean distance. Centroids are cluster modes (most frequent category per attribute). A partial reinitialization strategy (20% of particles reset every 5 stagnant iterations) prevents premature convergence.

**Parameters:** 80 particles, 300 iterations

| Metric | Value |
|---|---|
| Total Hamming distance | 11,591.0 |
| Adjusted Rand Index | **0.6060** |
| Silhouette Score | 0.2587 |
| Cluster 0 | 938 pts — 777 edible, 161 poisonous |
| Cluster 1 | 562 pts — 5 edible, 557 poisonous |

PSO-KModes achieved the highest ARI (0.606), outperforming all other clustering methods, by preserving the original categorical structure rather than projecting to continuous PCA space.

---

### Clustering Comparison Summary

| Algorithm | k | ARI ↑ | Silhouette ↑ | Fitness metric |
|---|---|---|---|---|
| PSO | 2 | 0.5854 | 0.2706 | Inertia |
| ABC | 5 (auto) | 0.2408 | **0.5203** | Davies-Bouldin |
| GWO | 2 | 0.0939 | 0.3465 | Inertia |
| PSO-KModes | 2 | **0.6060** | 0.2587 | Hamming distance |

> **ARI** (Adjusted Rand Index): measures how well clusters match true labels. 1.0 = perfect, 0.0 = random.  
> **Silhouette**: measures geometric cluster quality. 1.0 = perfectly separated, 0.0 = overlapping.

---

## Overall Summary

| Notebook | Algorithm | Task | Key Result |
|---|---|---|---|
| `abc_feature_selection.ipynb` | ABC | Feature Selection | 4/22 features → 99.82% accuracy (−81.8% features) |
| `pso_hyperparameter_tuning.ipynb` | PSO | Hyperparameter Tuning | 100% accuracy with auto-found RF hyperparameters |
| `pso_neural_network.ipynb` | PSO | NN Training (no backprop) | 98.09% test accuracy, 513 weights optimized by swarm |
| `swarm_clustering.ipynb` | PSO / ABC / GWO / PSO-KModes | Unsupervised Clustering | PSO-KModes best ARI (0.606); ABC best Silhouette (0.52) |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

All notebooks run on Google Colab. The dataset (`mushrooms.csv`) can be loaded from Google Drive or downloaded directly from the UCI ML Repository.

---

## References

- Karaboga, D. (2005). *An idea based on honey bee swarm for numerical optimization*. Technical Report TR06, Erciyes University.
- Kennedy, J. & Eberhart, R. (1995). *Particle swarm optimization*. ICNN.
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey wolf optimizer*. Advances in Engineering Software, 69, 46–61.
- UCI ML Repository — Mushroom Dataset: https://archive.ics.uci.edu/ml/datasets/mushroom
