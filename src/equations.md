$$
\mathbf{s}_i = 
\begin{bmatrix}
    \mathbf{t}_w \\
    \mathbf{A} \\
\end{bmatrix}^{(D_i=802) \times 1}
$$

$$
\mathbf{s}_o = \begin{bmatrix}
    \mathbf{p} \\
    \mathbf{T} \\
    \mathbf{M} \\
    \mathbf{T}_s \\
    \mathbf{T}_w \\
    \mathbf{q}_w \\
\end{bmatrix}^{(D_o=252840) \times 1 }
$$



$$
\mathbf{S}_i = \left[ \mathbf{s}_i^1 | \dots | \mathbf{s}_i^{N}\right]^{D_i\times N}
$$

$$
\mathbf{S}_o = \left[ \mathbf{s}_o^1 | \dots | \mathbf{s}_o^{N}\right]^{D_o\times N}
$$

$$
\mathbf{S}_i =\mathbf{U}_i\mathbf{\Sigma}_i\mathbf{V}_i^T
$$

$$
\mathbf{S}_o =\mathbf{U}_o\mathbf{\Sigma}_o\mathbf{V}_o^T
$$

$$
\mathbf{S}_i \approx \mathbf{\tilde U}_i^{N \times (k_i=5 \ll D_i)} \mathbf{\tilde\Lambda}_i^{(k_i=5 \ll D_i) \times N}
$$

$$
\mathbf{S}_o \approx \mathbf{\tilde U}_o^{N \times (k_o=10 \ll D_o)} \mathbf{\tilde\Lambda}_o^{(k_o=10 \ll D_o) \times N}
$$


$$
 \mathbf{S}_i = \mathbf{\tilde U}_i \mathbf{\tilde\Lambda}_i
  \Longleftrightarrow
 \mathbf{\tilde\Lambda}_i = \mathbf{\tilde U}_i^T \mathbf{S}_i
$$

$$
\mathbf{S}_o = \mathbf{\tilde U}_o \mathbf{\tilde\Lambda}_o
\Longleftrightarrow 
\mathbf{\tilde\Lambda}_o = \mathbf{\tilde U}_o^T \mathbf{S}_o 
$$

$$
\mathbf{\tilde\Lambda}_o = \mathcal{N}(\mathbf{\tilde\Lambda}_i)
$$

$$
 \mathbf{s}_i = \mathbf{\tilde U}_i \mathbf{\tilde\lambda}_i
  \Longleftrightarrow
 \mathbf{\tilde\lambda}_i = \mathbf{\tilde U}_i^T \mathbf{s}_i
$$

$$
 \mathbf{s}_o = \mathbf{\tilde U}_o \mathbf{\tilde\lambda}_o
  \Longleftrightarrow
 \mathbf{\tilde\lambda}_o = \mathbf{\tilde U}_o^T \mathbf{s}_o
$$

$$
\mathbf{\tilde\lambda}_o = \mathcal{N}(\mathbf{\tilde\lambda}_i)
$$

$$
\mathcal{L}_{MSE}(\mathbf{\tilde\Lambda}_o ,\mathcal{N}(\mathbf{\tilde\Lambda}_i))
$$