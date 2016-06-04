# COMPSCI 369 Exercises

### Find the eigenvalues of \(A=\begin{bmatrix}3&5\\1&-1\end{bmatrix}\).

The eigenvalues \(\lambda_{i}\) will satisfy \(det(A-\lambda_{i}\mathbf{I})=0\).  
\(\therefore det(\begin{bmatrix}3&5\\1&-1\end{bmatrix}-\lambda_{i}\begin{bmatrix}1&0\\0&1\end{bmatrix})=0\)  
\(\Rightarrow det(\begin{bmatrix}3&5\\1&-1\end{bmatrix}-\begin{bmatrix}\lambda_{i}&0\\0&\lambda_{i}\end{bmatrix})=0\)  
\(\Rightarrow det(\begin{bmatrix}3-\lambda_{i}&5\\1&-1-\lambda_{i}\end{bmatrix})=0\)  
\(\because det(\begin{bmatrix}a&b\\c&d\end{bmatrix})=ad-bc\)  
\(\therefore det(\begin{bmatrix}3-\lambda_{i}&5\\1&-1-\lambda_{i}\end{bmatrix})=(3-\lambda_{i})(-1-\lambda_{i})-(5)(1)=0\)  
\(\Rightarrow \lambda_{i}^2-2\lambda_{i}-8=0\)  
\(\because \text{Quadratic formula: }x=\Large\frac{-b\pm \sqrt{b^2-4ac}}{2a}\large\text{ and in this case }x=\lambda_{i}\)  
\(\Rightarrow \lambda_{i}=\Large\frac{-(-2)\pm \sqrt{(-2)^2-4(1)(-8)}}{2(1)}\)  
\(\Rightarrow \lambda_{i}=1\pm 3\)  
\(\therefore \lambda_{1}=1+3=4\)  
\(\therefore \lambda_{2}=1-3=-2\)
