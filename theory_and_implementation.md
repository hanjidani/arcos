# Theory and Implementation of the Risk-Difference Bound

This document outlines the theoretical foundation for bounding the change in model risk under data distribution shifts and its practical implementation in this project.

## 1. The Main Four-Term Decomposition

The core of our analysis is a theorem that decomposes the absolute difference in risk between a source model `Q` on a source distribution `P` and a target distribution `P_tilde`.

Let:
- `R_P(Q)`: The true risk of model `Q` on distribution `P`.
- `S`: The source training dataset, drawn from `P`.
- `\tilde{S}`: The target training dataset, drawn from `\tilde{P}`.
- `\widehat{R}_S(Q)`: The empirical risk of model `Q` on dataset `S`.

The theorem provides an upper bound on `|\Delta R| = |R_P(Q) - R_P(\tilde{Q})|`.

### Theorem: Risk-Difference Bound (Full)

Under covariate shift, the true risk difference is bounded by four key terms:

$$
|\Delta R| \le \underbrace{|R_P(Q) - \widehat{R}_S(Q)|}_{G_Q} + \underbrace{|R_{\tilde{P}}(\tilde{Q}) - \widehat{R}_{\tilde{S}}(\tilde{Q})|}_{G_{\tilde{Q}}} + \underbrace{|\widehat{R}_S(Q) - \widehat{R}_{\tilde{S}}(\tilde{Q})|}_{D_{Q,\tilde{Q}}} + \text{ShiftPenalty}
$$

Where:
- `G_Q`: The generalization gap of the source model.
- `G_{\tilde{Q}}`: The generalization gap of the target model.
- `D_{Q,\tilde{Q}}`: The empirical discrepancy between the two models on their respective training data.
- `ShiftPenalty`: A term that quantifies the severity of the distribution shift itself.

## 2. Defining the Shift Penalty with Distance Metrics

The `ShiftPenalty` term depends on the choice of distance metric used to compare the source distribution `P_X` and the target distribution `\tilde{P}_X`.

### 2.1. Wasserstein Bound

When using the Wasserstein-1 distance (`W_1`), the shift penalty is given by:

$$
\text{ShiftPenalty}_{W_1} = L_x(\phi) \cdot W_1(P_X, \tilde{P}_X)
$$

Where `L_x(\phi)` is the Lipschitz constant of the model's loss function with respect to the inputs.

By the triangle inequality, the true population distance `W_1(P_X, \tilde{P}_X)` can be related to the empirical distance `W_1(\widehat{P}_n, \widehat{\tilde{P}}_n)`:

$$
W_1(P_X, \tilde{P}_X) \le W_1(\widehat{P}_n, \widehat{\tilde{P}}_n) + \varepsilon_n
$$

Where `\varepsilon_n` is the statistical error term that arises from using a finite sample. This gives the full theoretical bound:

$$
|\Delta R|_{W_1} \le G_Q + G_{\tilde{Q}} + D_{Q,\tilde{Q}} + L_x(\phi) \cdot (W_1(\widehat{P}_n, \widehat{\tilde{P}}_n) + \varepsilon_n)
$$

### 2.2. MMD Bound

Similarly, when using Maximum Mean Discrepancy (MMD), the shift penalty is:

$$
\text{ShiftPenalty}_{MMD} \le \text{MMD}(P_X, \tilde{P}_X)
$$

And the empirical relationship is:

$$
\text{MMD}(P_X, \tilde{P}_X) \le \text{MMD}(\widehat{P}_n, \widehat{\tilde{P}}_n) + \varepsilon'_n
$$

Where `\varepsilon'_n` is the statistical error term, which converges at a rate of `O(n^{-1/2})` under standard assumptions.

This gives the full theoretical bound using MMD:

$$
|\Delta R|_{MMD} \le G_Q + G_{\tilde{Q}} + D_{Q,\tilde{Q}} + \text{MMD}(\widehat{P}_n, \widehat{\tilde{P}}_n) + \varepsilon'_n
$$

## 3. Practical Implementation in This Project

In our `run_adversarial_stress_test.py` script, we compute an *empirical version* of these bounds.

**Crucial Point on `\varepsilon_n` (Why it is Omitted):** The statistical error terms (`\varepsilon_n` and `\varepsilon'_n`) are omitted from the direct calculation and set to a placeholder value of 0. This is a standard practice in empirical studies for the following reasons:

1.  **For Wasserstein Distance:** The error term's convergence rate, `O(n^{-1/d})`, depends on the data's **intrinsic dimension `d`**. Estimating `d` for high-dimensional feature spaces is notoriously difficult and unreliable. Without a known `d`, this term cannot be computed.
2.  **For MMD:** While the rate `O(n^{-1/2})` is advantageously dimension-free, the full bound involves hidden constants related to the data distribution and the kernel. Calculating a precise numerical value for this term would require a secondary, complex statistical analysis (e.g., bootstrapping) which is beyond the scope of this primary experiment.

Therefore, the bounds we compute are practical, empirical estimates. Our code explicitly includes `eps_w1_placeholder` and `eps_mmd_placeholder` (set to 0) to transparently acknowledge where these non-computable terms exist in the full theory.

### Implemented Wasserstein Bound:

$$
\text{Bound}_{W1} = G_Q + G_{\tilde{Q}} + D_{Q,\tilde{Q}} + W_1(\text{images}_S, \text{images}_{\tilde{S}})
$$

### Implemented MMD Bound:

$$
\text{Bound}_{MMD} = G_Q + G_{\tilde{Q}} + D_{Q,\tilde{Q}} + \text{MMD}(\text{images}_S, \text{images}_{\tilde{S}})
$$

Where all terms are computed empirically from the source and target (adversarial) datasets. The final CSV report includes the results for both of these implemented bounds, allowing for a direct comparison of their tightness under adversarial shift.
