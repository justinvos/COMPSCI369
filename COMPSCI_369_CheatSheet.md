# COMPSCI 369 Cheatsheet

\(det(\begin{bmatrix}a&b\\ c&d\end{bmatrix})=ad-bc\)

### Eigenvalues
\(det(A-\lambda \mathbf I)=0\)

### Singular Value decomposition
\(A=UDV^T\)

### Principal component analysis
\(\Sigma=\frac{1}{n-1}AA^T\)

### QR decomposition
\(R=Q^TA\)

#### Gram-Schmidt process
\(q_{j}=\Large\frac{v_{j}}{|v_{j}|}\)  
\(v_{j}=a_{j}-\sum_{i=1}^{j-1}{(a_{j}^Tq_{i})q_i}\)
