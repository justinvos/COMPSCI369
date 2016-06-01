## Mathematical problems

A problem is **well-posed** if:
1. A solution exists.
2. The solution is unique.
3. A small change in the initial condition induces only a small change in the solution.

The **condition number** measures how sensitive a problem is to varying inputs. \(cond=\frac{|\text{relative change in solution}|}{|relative change in input data|}=\frac{\Delta y/y}{\Delta x/x}\).  
A problem is **ill-condtioned** if \(cond>1\).

An algorithm is **stable** if the result is relatively unaffected by perturbations during computation.  
An algorithm is **accurate** if the completed solution is close to the true solution of the problem.

## Finding roots

The **bisection method** is a slow-but-sure algorithm for finding a root of an equation. It involves iteratively narrowing a window containing the root.

#### Bisection algorithm
**Assuming:** *a and b are positions such that \(sign(f(a))\neq sign(f(b))\).*  
**Initialisation:**  
**Iterate until** \(f(a)-f(b)\approx 0\):  
\(m=(a+b)/2\)  
\(\text{if}\ sign(f(m))=sign(f(a))\ \text{then:}\)  
\(\quad a\leftarrow m\)  
\(else:\)  
\(\quad b\leftarrow m\)

The **Newton-Raphson method** is a faster algorithm for finding a root of an equation. It involves iteratively approximating the root using: \(x_{i+1}=x_i - \frac{f(x_i)}{f'(x_i)}\).
#### Newton-Raphson algorithm
**Assuming:** *x is a position and f is continuous and differentiable.*  
**Initialisation:**  
**Iterate until** \(x_{i}-x_{i-1}\approx 0\):  
\(x_{i+1}\leftarrow x_i-\frac{\Large f(x_i)}{f'(x_i)}\)  

## Linear algebra

The **identity matrix** is a square matrix with only 1's along the diagonal, denoted \(I_n\) where \(n\) is the size of the matrix.

The **determinant** of a matrix \(A\) is defined as \(det(A)=det(\begin{bmatrix}a&b\\ c&d\end{bmatrix})=ad-bc\).

A matrix is **invertible** (has an inverse) when \(det(A)\neq 0\).  
A matrix is **singular** (has no inverses) when \(det(A)=0\).

The **inverse** of a matrix is defined \(A^{-1}=\frac{\Large 1}{\Large det(A)}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}\).

An **eigenvector** is a non-zero vector such that \(Ae=\lambda e\) for some scalar \(\lambda\).  
The **eigenvalue** is the scalar \(\lambda\) corresponding to \(e\).  
The eigenvalues of \(A\) can be found using \(det(A-\lambda I)=0\).

The **magnitude** of a vector \(v\) is \(|v|=\sqrt{v_1^2+...+v_n^2}\).  
A vector is **normalised** when \(|v|=1\).

The **dot product** between two vectors is defined \(u\cdot v=\sum{u_i\times v_i}\).

Two vectors \(u\) and \(v\) are **orthogonal** if \(u\cdot v=0\).  
A set of vectors is **mutually orthogonal** when each vector is orthogonal to every other vector.  
A set of vectors is **orthonormal** when each vector is mutually orthogonal and each vector is normalised.

## Solving linear equations

A **linear systems of equations** \(Ax=b\) where \(A\) is the coefficient matrix, \(x\) is the solution vector and \(b\) is the constants matrix.  
This can be solved by finding \(A^{-1}\) and applying it to get \(x=A^{-1}b\).


If \(A\) is orthonormal, then \(A^T=A^{-1}\).


## Factorising matrices

**LU decomposition** is factorising \(A\) into the form \(A=LU\) where \(L\) is a lower triangular matrix and \(U\) is an upper triangular matrix.

**Singular Value Decomposition** is factorising \(A\) into the form \(A=UDV^T\) where \(U\) and \(V\) are orthogonal matrices and \(D\) is a diagonal matrix containing \(\sqrt{\lambda_i}\).

The **covariance matrix** of the matrix \(A\) is \(\Sigma=\frac{1}{n-1}AA^T\).

**QR decomposition** is factorising \(A\) into the form \(A=QR\) where \(Q\) is an orthogonal matrix and \(R\) is an upper triangular matrix.

## Genetics

**Deoxyribonucleic acid (DNA)** is one of the symbol sets containing genetic information. \(\text{DNA}=\{A,C,G,T\}\).

**Ribonucleic acid (RNA)** or **Messenger ribonucleic acid (mRNA)** is another one of the symbol sets containing genetic information. \(\text{RNA}=\{A,C,G,U\}\).

The **purines** are \(A\) and \(G\).  
The **pyramidines** are \(C\), \(T\) and \(U\).

**Proteins** are the twenty amino acids represented by the alphabet: \(\{A,R,N,D,C,E,Q,G,H,I,L,K,M,F,P,S,T,W,Y,V\}\). This is the entire English alphabet except for \(\{B,J,O,U,X,Z\}\).

**Translation** is the process of converting DNA into mRNA.  
**Transcription** is the process of converting mRNA into proteins.

A **codon** is a sequence of DNA of length 3.

**Recombination** is the process of mixing the paternal and maternal copies of the genetic information.

**Point mutations** are mutations to a single point in the genetic code.  
A **single nucleotide polymorphism** is where a single base is swapped in the child's genetic code.  
**Insertions** are when a sequence is added to an offspring's genetic code.  
**Deletions** are when a sequence is removed from its genetic code.  
**Gene duplication** is when a child inherits an extra copy of a whole gene.  
**Inversions** are when a sequence is reversed in the copying.
**Translocations** are when a sequence is copied out of order.

## Probability

A **stochastic process** is one where the results are non-deterministic, but still influenced by factors.

The **axioms of probability** are:
1. \(P(\Omega)=1\).
2. \(0\le P(A)\le 1\) for any \(A\subseteq \Omega\).
3. If \(A_1\),\(A_2\),... are mutually disjoint events (i.e. \(A_i\cap A_j=\emptyset\) if \(i\neq j\)) then \(P(\underset{i}{\LARGE\cup}A_i)=\underset{i}{\Large\sum}{P(A_i)}\).

The **conditional probability** is the probability of an event given another event.  
The probability of A given B is: \(P(A|B)=\large\frac{P(A\cap B)}{P(B)}\).

Two events are **independent** iff \(P(A\cap B)=P(A)\times P(B)\).

**Bayes' Theorem** states that: \(P(B|A)=\large\frac{P(A|B)\ P(B)}{P(A)}\).

### Random variables

A **random variable** is a variable whose value results from a random process.

The **expected value** of a random variable \(X\) is the mean \(\mu\), defined \(E[X]=\underset{x\in X}\sum{x\times P(x)}\).

The **variance** of a random variable is \(Var(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2\)

A **probability density function** returns the probability of a possible value.

**Marginalisation** is the process of finding \(P(x)\) by summing all the possible values of \(P(x,y)\) for \(y\) i.e. \(\underset{y\in Y}{\sum} P(x,y)\).

**Entropy** measures the unpredictability of a random variable. The entropy can be calculated using \(H(X)=-\underset{x\in X}\sum{P(x)log(P(x))}\).

### Probability distributions

A **Bernoulli distribution** can have the values 0 (failure) and 1 (success). It takes the probability of success \(p\) as a parameter. \(q\) is the probability of failure and is calculated \(q=1-p\).  
\(E[X]=p\)  
\(Var(X)=pq\)

A **Geometric distribution** is the number of Bernoulli trials that before the first success (can be inclusive or exclusive of the success trial). It only takes the probability of success \(p\).  
\({PDF}_{Geom}(x, p)=(1-p)^xp\)  
\(E[X]=\frac{q}{p}\)  
\(Var(X)=\frac{q}{p^2}\)

A **Binomial distribution** is the number of successes in \(n\) Bernoulli trials. Therefore it takes both the probability of success \(p\) and the number of trials \(n\).  
\({PDF}_{Bin}(x,n,p)=\frac{n!}{x!(n-x)!}p^x(1-p)^{n-p}\)  
\(E[X]=np\)  
\(Var(X)=npq\)

A **Poisson distribution** is used to model the number of rare events that occur in a period of time. It takes the event rate \(\lambda\).  
\({PDF}_{Poiss}(x,\lambda)=e^{-\lambda}{\large\frac{\lambda^x}{x!}}\)  
\(E[X]=\lambda\)  
\(Var(X)=\lambda\)

A **Uniform distribution** represents a model where all values are equally likely. It only takes the number of possibilities \(n\).  
\({PDF}_{U}(n)=1/n\)

A **Normal distribution** or a **Gaussian distribution** is Normally distributed along a bell curve with mean \(\mu\) and variance \(\sigma^2\).  
\({PDF}_(x,\mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}}exp\{-\frac{1}{2\sigma^2}(x-\mu)^2\}\)


An **Exponential distribution** models the time between rare events and therefore only accepts non-negative values. It takes the event rate \(\lambda\).  
\({PDF}_{Exp}(x, \lambda)=\lambda e^{-\lambda x}\)  
\(E[X]=\frac{1}{\lambda}\)  
\(Var(X)=\frac{1}{\lambda^2}\)

## Inference

The **likelihood** of observing a set of data \(D\) is \(P(D|\theta)\) where \(\theta\) is the set of parameters for our model. This can be calculated as \(P(D|theta)=\underset{i}\prod{P(D_i|\theta)}\).

The **prior distribution** is the distribution before making any observations, denoted \(P(\theta)\).

The **posterior distribution** is the distribution estimated from the observed data, denoted \(P(\theta|D)\). It can be calculated using \(P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}\) where \(P(D)\) is the normalisation constant.

The **normalisation constant** is denoted \(P(D)\).

## Alignment
Two sequence regions are **homologous** that share a common ancestry. The level of similarity depends on how recently they shared a common ancestor.

**Orthology** occurs when two genes are separated by a speciation event and evolve independently from there on.  
**Paralogy** occurs when a region of the genome is duplicated in the same genome (a duplication event) and they evolve in parallel in the same genome. The two copies are said to be paralogs.

**Pairwise alignment** is the problem of optimally aligning two sequences.

A **linear gap penalty** is defined as \(\gamma(k)=-dk\) where \(d>0\) and \(k\) is the gap length.  
An **affine gap penalty** is defined as \(\gamma(k)=-d-(k-1)e\) where \(d>e>0\) and \(k\) is the gap length. \(d\) is the gap open penalty and \(e\) is the gap extension penalty.

**Global alignment** refers to the optimal alignment of whole sequences.  
The **Needleman-Wunsch algorithm** can be used to find a global alignment.

**Local alignment** refers to the optimal alignment of some subsequences within each sequence.  
The **Smith-Waterman algorithm** can be used to find a local alignment.

### Multiple sequence alignment

**Multiple sequence alignment (MSA)** is the alignment of more than two sequences.

**Progressive alignment** is a widely-used heuristic technique for finding a good enough multiple sequence alignment. It involves iteratively using pairwise alignment on two sequences and using the aligned sequences with others.

**Unweighted pair group method using arithmetic averages (UPGMA)** is a method to decide the ordering of pairwise alignments in a multiple sequence alignment.  
The distance between two clusters is the average distance between all pairs between clusters: \(d_{ij}\frac{1}{|C_i||C_j|} \sum_{x\in C_i,y\in C_j} d_{xy}\) where \(|C|\) is the number of sequences in the cluster and \(d_{xy}\) is a distance function/matrix between two clusters.

## Hidden Markov Models

A **hidden markov model (HMM)** is just a Markov model with an extra symbol sequence where each symbol \(x\) is dependent on the current state \(\pi\).

The **Viterbi algorithm** is a dynamic programming algorithm for finding the most likely sequence of hidden states from a sequence of symbols. It involves building a matrix of maximum scores (the maximum path score to that node) and a matrix of the previous state that made the maximum path score, then backtracking using the previous state matrix from the node with the highest maximum score results in the most likely sequence. \(Sc_{i\ j}=P_{symbol}(x_j|S_j)\times \underset{r}max(Sc_{r\ j-1}\times P_{transition}(Sc_{i\ j}|Sc_{r\ j-1}))\) where \(Sc\) is the score matrix, \(P_{symbol}(A|B)\) is probability of the symbol \(A\) given the state \(B\), \(x\) is the inputted symbol sequence.

## Parsimony

The **maximum parsimony tree** is the tree that requires the fewest changes along it to explain all sequences.

The **parsimony algorithm** involves finding the set of possible ancestor bases at each site and when an empty set is found, the cost is incremented.

The **weighted parsimony algorithm** uses a separate score cost for each base to be the ancestor root at that site.

**Branch and bound** algorithms involve abandoning possibilities once they are of the same or worse score than the current best possibility. This is based on the idea that the score will only either stay the same or get worse with more additions.

**Heuristic search** gives up on finding the maximum parsimony tree, and attempts to find a good-enough tree in a much more efficient manner.

**Subtree prune regraft (SPR)** involves choosing random edges and reattaching them at other random edges to reconstruct the tree.

**Probems with parsimony**:
1. Ignores hidden mutations, and therefore underestimates branch lengths.
2. The more data on mutations can result in higher certainty in the wrong topology, because the randomness of mutations can be mistaken for homology.
