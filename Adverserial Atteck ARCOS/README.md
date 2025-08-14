# Adversarial Robustness with Theoretical Bounds

This repository implements an improved framework for evaluating adversarial robustness using theoretical bounds based on Wasserstein distances, Lipschitz constants, and generalization gaps.

## Overview

The code implements the theoretical framework from the methodology, providing:
- **Linear head experiments** on frozen features (`run_adversarial_experiment.py`)
- **End-to-end training** with adversarial data (`run_adversarial_end_to_end.py`)
- **Enhanced utilities** for bound computation (`adversarial_utils.py`)

## Key Improvements Made

### 1. **Fixed Critical Bugs**
- ✅ Resolved import/name inconsistencies across files
- ✅ Fixed function naming (`calculate_wasserstein1_distance`)
- ✅ Fixed gradient flow semantics with automatic feature freezing

### 2. **Enhanced Lipschitz Estimation**
- ✅ **Loss-based gradients** instead of feature sum gradients (more theoretically sound)
- ✅ **Jacobian norm aggregation** as alternative method
- ✅ Temperature scaling support for better numerical stability

### 3. **Improved Output Distance Metrics**
- ✅ **KL divergence** between softmax outputs (replaces raw L2)
- ✅ **Temperature-scaled softmax** for better scale awareness
- ✅ **Automatic temperature calibration** using ECE/ACE minimization
- ✅ Maintains backward compatibility with legacy L2 distance

### 4. **Efficient Wasserstein Computation**
- ✅ **Sinkhorn algorithm** for large datasets (O(n²) vs O(n³))
- ✅ **Feature normalization/whitening** for numerical stability
- ✅ **Automatic fallback** to exact computation for small datasets
- ✅ **Unequal sample size support** with weights
- ✅ **Convergence checks** and configurable parameters

### 5. **Multidimensional Adversarial Search**
- ✅ **PCA-based shifts** along principal directions
- ✅ **OT transport vector** directions
- ✅ **Gradient-based optimization** to maximize bound-to-risk ratio
- ✅ Replaces simple single-dimension shifts

### 6. **Enhanced Diagnostics & Reproducibility**
- ✅ **Comprehensive metrics** logging (bound components, tightness)
- ✅ **Reproducibility** with proper seeding
- ✅ **Validation splits** for hyperparameter tuning
- ✅ **Early stopping** to prevent overfitting

### 7. **Attack Calibration & Evaluation**
- ✅ **Multiple epsilon values** for comprehensive evaluation
- ✅ **Proper model.eval()** during adversarial generation
- ✅ **AutoAttack integration** for sanity checks
- ✅ **Multiple attack types** (PGD, FGSM)

### 8. **Memory & Performance Optimizations**
- ✅ **Chunked distance computation** to avoid OOM
- ✅ **Efficient Sinkhorn Wasserstein** for large datasets
- ✅ **Batch processing** for Lipschitz estimation

### 9. **Temperature Calibration & Model Calibration**
- ✅ **Expected Calibration Error (ECE)** computation
- ✅ **Adaptive Calibration Error (ACE)** for robustness
- ✅ **Automatic temperature optimization** on validation set
- ✅ **Temperature sweep** across τ ∈ {0.5, 1, 2, 4}

### 10. **Per-Bucket Tightness Analysis**
- ✅ **Class-based buckets** for per-class analysis
- ✅ **Confidence-based buckets** for uncertainty analysis
- ✅ **Entropy-based buckets** for prediction complexity
- ✅ **Perturbation norm buckets** for attack strength analysis
- ✅ **Reveals bound tightness** on hard vs. easy examples

### 11. **Gradient Flow & Space Consistency**
- ✅ **Automatic feature freezing** during W₁ computation
- ✅ **EMA teacher creation** for stable regularizers
- ✅ **Consistent space semantics** (features vs. images)
- ✅ **Prevents accidental training** of measured spaces

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install AutoAttack for comprehensive evaluation
pip install autoattack
```

## Usage

### Linear Head Experiment

```bash
python run_adversarial_experiment.py \
    --search_budget 1000 \
    --shift_magnitude 0.5 \
    --epochs 50 \
    --seed 42
```

**What it does:**
1. Loads pretrained feature extractor
2. Trains linear classifiers on frozen features
3. Searches for worst-case adversarial shifts
4. Computes bound tightness and diagnostics

### End-to-End Experiment

```bash
python run_adversarial_end_to_end.py \
    --epochs 10 \
    --batch_size 128 \
    --attack_eps 0.1 \
    --attack_alpha 0.01 \
    --attack_steps 10 \
    --seed 42
```

**What it does:**
1. Trains models from scratch on clean data
2. Generates adversarial datasets with multiple epsilon values
3. Trains new models on adversarial data
4. Computes comprehensive bound analysis
5. Runs AutoAttack evaluation (if available)

## Output Files

### `results/adversarial_tightness_results.csv`
- Worst-case shift analysis
- Bound components breakdown
- Tightness metrics

### `results/adversarial_end_to_end_comprehensive.csv`
- Multi-epsilon evaluation results
- **Two tightness metrics**: Model comparison vs. distribution shift
- Bound components breakdown (four-term decomposition)
- Temperature calibration and ECE metrics
- AutoAttack comparison
- Space information (W1 on images, Lipschitz wrt images)

## Key Metrics Explained

### **Bound Components**
- **g_q, g_q_tilde**: Generalization gaps $G_Q, G_{\tilde Q}$
- **l_x_q, l_x_q_tilde**: Lipschitz constants $L_x^h(f_Q), L_x^h(f_{\tilde Q})$
- **w1_dist**: Wasserstein-1 distance $W_1^h$ (computed on detached features)
- **output_dist**: KL divergence between outputs (surrogate for $D_{Q,\tilde Q}$)

### **Theoretical vs. Empirical Correspondence**

| **Theory** | **Implementation** | **Notes** |
|------------|-------------------|-----------|
| $W_1(P, \tilde P)$ | `calculate_wasserstein1_distance_sinkhorn()` | Sinkhorn on detached features |
| $L_x(f)$ | `estimate_lipschitz_constant_loss_based()` | Loss gradients wrt input space |
| $D_{Q,\tilde Q}$ | `calculate_output_distance_kl()` | Temperature-scaled KL surrogate |
| $\xi_n$ | Sampling error | Captured in convergence analysis |
| $\varepsilon_n^h$ | Feature space sampling | Sinkhorn tolerance + convergence |

### **Space Consistency**
- **Linear Head Experiment**: W₁ computed on features, Lipschitz estimated wrt features
- **End-to-End Experiment**: W₁ computed on images, Lipschitz estimated wrt images
- **Output Distance**: Always computed in logit/softmax space (model outputs)

### **Gradient Flow Semantics**
- **W₁ Computation**: Features are automatically detached (`freeze_features=True`) to prevent gradients from flowing through the feature map h
- **EMA Teacher**: For end-to-end training, use `create_ema_teacher()` to create a frozen copy of h for stable W₁ computation
- **Regularizer Stability**: W₁(h(S), h(S̃)) keeps the regularizer stable instead of chasing a moving embedding
- **Training Modes**: 
  - Use `model.eval()` during adversarial generation to avoid BN/Dropout randomness
  - Use `model.train()` during model updates
  - W₁ computation always uses frozen features regardless of training mode

### **Tightness Analysis**
The implementation computes **two different tightness metrics** to properly analyze the bound:

#### **1. Model Comparison Tightness (Primary)**
- **Formula**: `bound / delta_r_model` where `delta_r_model = |R_P(Q) - R_P(Q̃)|`
- **Meaning**: How tight the four-term bound is for comparing different models on the same distribution
- **Target**: Values closer to 1 indicate the bound is tight for model comparison

#### **2. Distribution Shift Tightness (Secondary)**
- **Formula**: `(L_x(Q) + L_x(Q̃)) * W1 / delta_r_shift` where `delta_r_shift = |R_P(Q) - R_P̃(Q)|`
- **Meaning**: How well the Lipschitz * W1 term captures distribution shift effects
- **Target**: Values closer to 1 indicate the Lipschitz assumption holds well

#### **Per-Bucket Analysis**
- **Class-based**: Per-class tightness to identify problematic classes
- **Confidence-based**: Tightness across uncertainty levels
- **Entropy-based**: Tightness across prediction complexity
- **Perturbation-based**: Tightness across attack strengths

**Theoretical Interpretation:**

**Model Comparison Tightness:**
- **Tightness ≈ 1**: The four-term bound is nearly tight, theoretical assumptions hold well
- **Tightness >> 1**: The bound is loose, suggesting:
  - Lipschitz constants may be overestimated
  - Wasserstein distance may not capture the true distribution shift
  - Output distance surrogate (KL) may not align with theoretical assumptions

**Distribution Shift Tightness:**
- **Tightness ≈ 1**: The Lipschitz * W1 term well-captures distribution shift effects
- **Tightness >> 1**: The Lipschitz assumption may be too conservative for the data

**Training Dynamics:**
- **Tightness improves with training**: Indicates better alignment between empirical and theoretical assumptions
- **Consistent tightness across buckets**: Suggests the bound generalizes well across different example types

### **Risk Metrics**
- **Delta R**: True risk change between distributions
- **Expected Loss**: Cross-entropy loss for bound analysis
- **Classification Error**: 0-1 risk for evaluation

## Advanced Usage

### Custom Feature Extractors

```python
from adversarial_utils import estimate_lipschitz_constant_loss_based

# Use your own feature extractor
lipschitz = estimate_lipschitz_constant_loss_based(
    your_model, data_loader, device, temperature=0.1
)
```

### Custom Distance Metrics

```python
from adversarial_utils import calculate_wasserstein1_distance_sinkhorn

# Compute Wasserstein distance with custom parameters
w1_dist = calculate_wasserstein1_distance_sinkhorn(
    X, Y, epsilon=0.01, max_iter=200
)
```

### Temperature Scaling

```python
from adversarial_utils import calculate_output_distance_kl

# Use temperature scaling for better numerical stability
kl_dist = calculate_output_distance_kl(
    model_q, model_q_tilde, features, device, temperature=0.1
)
```

## Theoretical Background

The framework implements the theoretical bound from the methodology, which provides a four-term decomposition for the risk difference between models under distribution shift.

### **Theorem 1: Risk Difference Bound**

**Assume covariate shift**: $P(y\!\mid\!x)=\tilde P(y\!\mid\!x)$ and only the marginals differ $P_X\neq \tilde P_X$. The Lipschitz constant is with respect to the metric used in $W_1$ on $\mathcal X$ (or on $h(\mathcal X)$ if working in feature space).

For any models $Q, \tilde Q$ and distributions $P, \tilde P$:

$$
|R_P(Q) - R_{\tilde P}(\tilde Q)| \leq G_Q + G_{\tilde Q} + \big(L_x(f_Q) + L_x(f_{\tilde Q})\big) \cdot W_1(P, \tilde P) + D_{Q,\tilde Q}
$$

**Where:**
- **G_Q, G_Q̃**: Generalization gaps
- **L_x^Q, L_x^Q̃**: Lipschitz constants with respect to the input space
- **W_1(P, P̃)**: Wasserstein-1 distance between distributions
- **D(f_Q, f_Q̃)**: Output distance (model change term)

**Definitions:**
$G_Q := |R_P(Q)-\widehat R_S(Q)|,\; G_{\tilde Q}:=|\widehat R_{\tilde S}(\tilde Q)-R_{\tilde P}(\tilde Q)|,\; D_{Q,\tilde Q}:=|\widehat R_S(Q)-\widehat R_{\tilde S}(\tilde Q)|.$

### **Proof Sketch:**

**Step 1:** Apply triangle inequality to decompose the risk difference:
$$
|R_P(Q) - R_{\tilde P}(\tilde Q)| \leq |R_P(Q) - \widehat R_S(Q)| + |\widehat R_S(Q) - \widehat R_{\tilde S}(\tilde Q)| + |\widehat R_S(\tilde Q) - R_{\tilde P}(\tilde Q)|
$$

**Step 2:** The middle term is bounded by the data shift:
$$
\big|\widehat R_S(Q)-\widehat R_{\tilde S}(Q)\big|
\;\le\; L_x(f_Q)\, W_1(\widehat P_n,\widehat{\tilde P}_n)\;+\;\xi_n,
$$
where $\xi_n=O_p(n^{-1/2})$ under bounded loss/variance. This moves the final result from "deterministic" to "with high probability".

**Step 3:** Apply K-R duality and Lipschitz assumptions to complete the bound.

### **Feature-Space Version (Implementation)**

When computing $W_1$ on $h(\mathcal X)$, we use feature-space constants:

$$
L_x^h(f):=\sup_{x}\|\nabla_x(\ell(f(x),y)\circ h)\|,\quad
W_1^h:=W_1\big(h_\#\widehat P_n,\,h_\#\widehat{\tilde P}_n\big).
$$

**Corollary (Feature-Space):** With probability $\ge 1-\delta$ over draws of $S,\tilde S$:

$$
|\Delta R| \;\le\;
G_Q+G_{\tilde Q}
+\big(L_x^h(f_Q)+L_x^h(f_{\tilde Q})\big)\,W_1^h
+L_x^h(f_{\tilde Q})\,\varepsilon_n^h
+L_\ell\cdot \tfrac1n\!\sum_{i=1}^n\!\|f_Q(\tilde x_i)-f_{\tilde Q}(\tilde x_i)\|
+\xi_n
$$

where $W_1^h=W_1(h_\#\widehat P_n, h_\#\widehat{\tilde P}_n)$ computed by Sinkhorn, and $\varepsilon_n^h,\xi_n$ capture sampling errors.

**Sampling Terms:**
- $\xi_n = O_p(n^{-1/2})$: Label sampling error under bounded loss/variance
- $\varepsilon_n^h = O_p(n^{-1/2})$: Feature space sampling error in $h(\mathcal X)$

### **Implementation Notes:**

- **Wasserstein distance**: Sinkhorn $W_1$ on **detached** features $h(x)$; weights uniform by default; unequal sizes supported
- **Output distance**: Temperature-scaled KL on softmax outputs (used for diagnostics and as an optional regularizer)
- **Lipschitz surrogate**: Loss-gradient norms (optionally Jacobian norms) wrt the space where $W_1$ is computed
- **Reproducibility**: Pinned deps (Torch/Numpy/etc.) for consistent runs

**Note on KL vs. ℓ₂**: The implementation uses temperature-scaled KL between softmax outputs as a **surrogate** for the ℓ₂-based theoretical bound. While KL provides better empirical properties, theoretical control via KL would use different constants (e.g., via smoothness of the log-partition and Pinsker-type bounds).

### **Mathematical Properties & Convergence**

**Wasserstein Properties:**
- **Symmetry**: $W_1(S, \tilde S) = W_1(\tilde S, S)$
- **Triangle Inequality**: $W_1(S, U) \leq W_1(S, T) + W_1(T, U)$ (within Sinkhorn tolerance)
- **Two-point Case**: $W_1(\{a\}, \{b\}) = \|h(a) - h(b)\|_2$ for single samples

**Convergence Guarantees:**
- **Sinkhorn Convergence**: Dual residual-based stopping criterion with configurable tolerance
- **Sampling Error**: $\xi_n = O_p(n^{-1/2})$ under bounded loss and variance assumptions
- **Feature Space**: $\varepsilon_n^h$ captures sampling error in the feature space $h(\mathcal X)$

**Lipschitz Estimation:**
- **Gradient-based**: $\|\nabla_x \ell(f(x), y)\|_2$ provides upper bound on local Lipschitz constant
- **Jacobian-based**: Alternative method using $\sum_c \|\nabla_x f^{(c)}(x)\|_2$ aggregation
- **Finite Difference Validation**: Cross-validated against finite difference approximations

### **ARCOS Objective (Adversarial Regularization)**

The theoretical framework leads to the following adversarial regularization objective:

$$
\mathcal{L}_{\text{ARCOS}} = \mathcal{L}_{\text{standard}} + \lambda_W \cdot W_1^h + \lambda_L \cdot L_x^h(f_{\tilde Q}) \cdot \varepsilon_n^h + \lambda_D \cdot D_{Q,\tilde Q}
$$

where:
- $\lambda_W$: Weight for Wasserstein regularization
- $\lambda_L$: Weight for Lipschitz regularization  
- $\lambda_D$: Weight for output distance regularization
- Constants are consistent with where $W_1$ is computed (use $L_x(f_{\tilde Q})$ or $L_x^h(f_{\tilde Q})$ accordingly)

## Performance Tips

1. **Use Sinkhorn Wasserstein** for datasets > 1000 samples
2. **Enable early stopping** to prevent overfitting
3. **Use validation splits** for hyperparameter tuning
4. **Chunk large distance matrices** to avoid memory issues
5. **Set proper seeds** for reproducible results

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce batch size or use chunked computation
3. **Numerical instability**: Use temperature scaling and feature normalization
4. **Slow Wasserstein**: Use Sinkhorn for large datasets

### Performance Optimization

1. **GPU acceleration**: Ensure CUDA is available
2. **Batch processing**: Use appropriate batch sizes
3. **Early stopping**: Prevent unnecessary training
4. **Efficient algorithms**: Use Sinkhorn over exact Wasserstein

## Contributing

When contributing:
1. Maintain backward compatibility
2. Add comprehensive tests
3. Update documentation
4. Follow the established code style

## Citation

If you use this code in your research, please cite the original methodology paper and mention the improvements made in this implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
