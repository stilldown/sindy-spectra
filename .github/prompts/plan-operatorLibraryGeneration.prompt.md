# Mixed-Domain Operator Library Generation Plan

This plan outlines the procedure for constructing a sparse operator library for N-dimensional control variables ($c$) and M-dimensional observation variables ($x$, e.g., wavelengths), utilizing a mixed-domain strategy to capturing both linear spectral dynamics and nonlinear physical laws.

## Core Philosophy
**"Select features in the frequency domain ($\omega$), but modulate coefficients in the physical control domain ($c$)."**

## Phase 0: Initialization & Mixed-Domain Transform
1.  **Input**: Raw data tensor $S(c_1, \dots, c_N, x)$.
    *   $c$-axis: Physical concentration/control grid (potentially non-uniform).
    *   $x$-axis: Uniform observation grid (e.g., wavelength).
2.  **Transform**: Apply FFT **only** along the $x$-axis.
    $$ \hat{S}(c, \omega) = \mathcal{F}_x \{ S(c, x) \} $$
3.  **Pre-computation**: Generate frequency vector $\mathbf{k}_x = i\omega$.

## Phase 1: Linear Spectral Basis Generation
Generate linear derivative terms directly in the frequency domain via multiplication.
For each order $n = 0, 1, 2, \dots$:
   $$ \text{LinearTerms}_n = (i\omega)^n \odot \hat{S}(c, \omega) $$
   *   $n=0$: $\hat{S}$ (The state itself)
   *   $n=1$: $\widehat{\partial_x S}$ (First derivative wrt $x$)
   *   $n=2$: $\widehat{\partial_x^2 S}$ (Diffusion/Curvature wrt $x$)

## Phase 2: Nonlinear Basis Generation ("The Round Trip")
Generate nonlinear terms (e.g., $S^2$, $(S_x)^2/S$) that cannot be represented by simple spectral multiplication.
1.  **Inverse Transform**: Bring necessary terms back to the time/spatial domain.
    $$ S(c, x) = \mathcal{F}^{-1}_x \{ \hat{S} \} $$
    $$ S_x(c, x) = \mathcal{F}^{-1}_x \{ (i\omega)\hat{S} \} $$
2.  **Calculate Functionals**: Compute nonlinearities in the physical domain.
    *   $F_1 = S^2$ (Hadamard product)
    *   $F_2 = |S|^2$ (Energy)
    *   $F_3 = (S_x)^2 / (S + \epsilon)$ (Log-derivative squared)
    *   $F_4 = x \cdot S_x$ (Euler/Scaling term)
3.  **Forward Transform**: Bring results back to the frequency domain.
    $$ \widehat{F_j} = \mathcal{F}_x \{ F_j \} $$
    *Output*: A set of nonlinear frequency tensors $\{\widehat{F_1}, \widehat{F_2}, \dots\}$.

## Phase 3: Concentration Derivative Generation
Generate dynamics regarding the control variables $c$.
1.  **Operation**: Compute gradients along the $c$-axis of the mixed-domain tensor $\hat{S}(c, \omega)$.
2.  **Method**:
    *   If $c$ is uniform: Use finite differences.
    *   If $c$ is non-uniform: Use `np.gradient` or NUFFT-based separate differentiation.
    $$ \widehat{\partial_{c_k} S} \approx \frac{\Delta \hat{S}}{\Delta c_k} $$
    *Output*: A set of concentration derivative tensors $\{\widehat{\partial_{c_1} S}, \dots\}$.

## Phase 4: Assembly & Control Modulation
Combine spectral/dynamic features ($B$) with control state features ($P(c)$).
1.  **Base Features ($B$)**: Collect all outputs from Phases 1, 2, and 3.
    $$ \mathbf{B} = \{ (i\omega)^n\hat{S}, \widehat{F_j}, \widehat{\partial_{c_k}S} \} $$
2.  **Control Polynomials ($P$)**: Construct polynomial basis for control variables.
    $$ \mathbf{P}(c) = \{ 1, c_1, c_1^2, c_1 c_2, \dots \} $$
    *   Note: These are broadcasted to match the $\omega$ dimension.
3.  **Tensor Product**: Form the final library $\Theta$.
    For every $B \in \mathbf{B}$ and $P \in \mathbf{P}$:
    $$ \Theta_{col} = P(c) \odot B(c, \omega) $$

## Phase 5: Sparse Regression (Spectral SINDy)
Solve the implicit equation $\Theta \xi = 0$ in the frequency domain.
1.  **Flatten**: Reshape $\Theta$ to $(N_{total}, N_{features})$.
    *   Optional: Subsample along the $\omega$ axis (keep high-energy modes) to reduce $N_{total}$.
2.  **Regress**: Apply Sparse Regression (STLSQ/SR3) to find the nullspace vector $\xi$.
3.  **Interpret**: Map $\xi$ back to the physical terms (e.g., "$c_1 \cdot \partial_x S + \partial_{c_1} S = 0$").

## Summary
1.  **FFT ($x$)** $\to$ Mixed Domain
2.  **Mult ($(i\omega)^n$)** $\to$ Linear Spectral Terms
3.  **IFFT $\to$ Calc $\to$ FFT** $\to$ Nonlinear Spectral Terms
4.  **Grad ($c$)** $\to$ Concentration Dynamics
5.  **Mult ($P(c)$)** $\to$ Modulation
6.  **Solve** $\to$ Discovery
