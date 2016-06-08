# COMPSCI 369 Notes
## Linear Systems and Root Finding
### Mathematical problems

A problem is **well-posed** if:
1. A solution exists.
2. The solution is unique.
3. A small change in the initial condition induces only a small change in the solution.

The **condition number** measures how sensitive a problem is to varying inputs. $cond=\frac{|\text{relative change in solution}|}{|relative change in input data|}=\frac{\Delta y/y}{\Delta x/x}$.  
A problem is **ill-condtioned** if $cond>1$.

An algorithm is **stable** if the result is relatively unaffected by perturbations during computation.  
An algorithm is **accurate** if the completed solution is close to the true solution of the problem.

### Finding roots

The **bisection method** is a slow-but-sure algorithm for finding a root of an equation. It involves iteratively narrowing a window containing the root.

#### Bisection algorithm
**Assuming:** *a and b are positions such that $sign(f(a))\neq sign(f(b))$.*  
**Iterate until** $f(a)-f(b)\approx 0$:  
$m=(a+b)/2$  
$\text{if}\ sign(f(m))=sign(f(a))\ \text{then:}$  
$\quad a\leftarrow m$  
$else:$  
$\quad b\leftarrow m$

The **Newton-Raphson method** is a faster algorithm for finding a root of an equation. It involves iteratively approximating the root using: $x_{i+1}=x_i - \frac{f(x_i)}{f'(x_i)}$.
#### Newton-Raphson algorithm
**Assuming:** *x is a position and f is continuous and differentiable.*  
**Iterate until** $x_{i}-x_{i-1}\approx 0$:  
$x_{i+1}\leftarrow x_i-\frac{\Large f(x_i)}{f'(x_i)}$  

### Vectors

The **magnitude** of a vector $v$ is $|v|=\sqrt{v_1^2+...+v_n^2}$.  
A vector is **normalised** when $|v|=1$.

The **dot product** between two vectors is defined $u\cdot v=\sum{u_i\times v_i}$.

Two vectors $u$ and $v$ are **orthogonal** if $u\cdot v=0$.  
A set of vectors is **mutually orthogonal** when each vector is orthogonal to every other vector.  
A set of vectors is **orthonormal** when each vector is mutually orthogonal and each vector is normalised.

### Matrices

The **identity matrix** is a square matrix with only 1's along the diagonal, denoted $\mathbf I_n$ where $n$ is the size of the matrix.

The **determinant** of a matrix $A$ is defined as $det(A)=det(\begin{bmatrix}a&b\\ c&d\end{bmatrix})=ad-bc$, also denoted $|A|$.

A matrix is **invertible** (has an inverse) when $det(A)\neq 0$.  
A matrix is **singular** (has no inverses) when $det(A)=0$.

The **inverse** of a matrix is defined $A^{-1}=\frac{\Large 1}{\Large det(A)}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$.

$A\times A^{-1}=\mathbf I$.

An **eigenvector** is a non-zero vector such that $Ae=\lambda e$ for some scalar $\lambda$.  
The **eigenvalue** is the scalar $\lambda$ corresponding to $e$.  
The eigenvalues of $A$ can be found using $det(A-\lambda \mathbf I)=0$.

### Solving linear equations

A **linear systems of equations** $Ax=b$ where $A$ is the coefficient matrix, $x$ is the solution vector and $b$ is the constants matrix.  
This can be solved by finding $A^{-1}$ and applying it to get $x=A^{-1}b$.

A system is **under-determined** if there are less equations ($m$ rows in $A$) than unknowns ($n$ columns in $A$) i.e. $m<n$.  
A system is **over-determined** if there are more equations ($m$ rows in $A$) than unknowns ($n$ columns in $A$) i.e. $m>n$.

A **diagonal matrix** is a matrix where $A_{ij}=0$ when $i\neq j$.  
The inverse of a diagonal matrix is $A^{-1}_{ij}=1/A_{ij}$ when $i=j$.

A **triangular matrix** is a matrix is where $A_{ij}=0$ when $i>j$ in upper triangular matrices or when $i<j$ in lower triangular matrices.


**Gaussian elimination** is a process that involves transforming $Ax=b$ into the equivalent equation $Cx=d$ where C is triangular.

If $A$ is orthonormal, then $A^T=A^{-1}$.


### Factorising matrices

**LU decomposition** is factorising $A$ into the form $A=LU$ where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix.

**Singular Value Decomposition** is factorising $A$ into the form $A=UDV^T$ where $U$ and $V$ are orthogonal matrices containing the column eigenvectors of $A^TA$ and $AA^T$ respectively and $D$ is a diagonal matrix containing the singular values of both $A^TA$ and $AA^T$.

**Singular values** are defined as $\sigma=\sqrt{\lambda}$.

The **condition number of a matrix** is defined $cond(A)=\Large\frac{\sigma_{max}}{\sigma_{min}}$.

The **covariance matrix** of the matrix $A$ is defined $\Sigma=\frac{1}{n-1}AA^T$.

### Least sqaures

In linear regression, finding a $u$ that satisfies $Au=b$ is usually impossible, so we attempt to find the best solution $u^*$.

The **error** or **residual** is defined as $e=b-Au$.  
The **total square error** is the sum of squared errors $E(u)=\sum{e_{i}^2}=\sum{e_{i}\cdot e_{i}}=e^T\cdot e$.  
$\therefore E(u)=(b-Au)^T(b-Au)$  
The best solution $u^*$ minimises $E(u)$, to find this we can differentiate and solve for $0$.  
This results in the **normal equation** which is $A^TAu=A^Tb$.

**QR decomposition** is factorising $A$ into the form $A=QR$ where $Q$ is an orthogonal matrix produced by the Gram-Schmidt process and $R$ is an upper triangular matrix $R=Q^TA$.

The **Gram-Schmidt** process involves finding the orthonormal columns $q_{1},\cdots,q_{n}$ of $Q$ from the columns $a_{1},\ldots,a_{n}$ of $A$.  
$q_{j}=\Large\frac{v_{j}}{|v_{j}|}$ where $v_{j}=a_{j}-\sum_{i=1}^{j-1}{(a_{j}^Tq_{i})q_i}$.

## Sampling and Markov models

### Probability

A **stochastic process** is one where the results are non-deterministic, but still influenced by factors.

The **axioms of probability** are:
1. $P(\Omega)=1$.
2. $0\le P(A)\le 1$ for any $A\subseteq \Omega$.
3. If $A_1$,$A_2$,... are mutually disjoint events (i.e. $A_i\cap A_j=\emptyset$ if $i\neq j$) then $P(\underset{i}{\LARGE\cup}A_i)=\underset{i}{\Large\sum}{P(A_i)}$.

The **conditional probability** is the probability of an event given another event.  
The probability of A given B is: $P(A|B)=\large\frac{P(A\cap B)}{P(B)}$.

Two events are **independent** iff $P(A\cap B)=P(A)\times P(B)$.

**Bayes' Theorem** states that: $P(B|A)=\large\frac{P(A|B)\ P(B)}{P(A)}$.

Using **log-likelihood** rather than likelihood makes it easier to algebraically calculate and helps to avoid numerical under-flow.

### Random variables

A **random variable** is a variable whose value results from a random process.

The **expected value** of a random variable $X$ is the mean $\mu$, defined $E[X]=\underset{x\in X}\sum{x\times P(x)}$.

The **variance** of a random variable is $Var(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2$

A **probability density function** returns the probability of a possible value.

**Marginalisation** is the process of finding $P(x)$ by summing all the possible values of $P(x,y)$ for $y$ i.e. $\underset{y\in Y}{\sum} P(x,y)$.

**Entropy** measures the unpredictability of a random variable. The entropy can be calculated using $H(X)=-\underset{x\in X}\sum{P(x)log(P(x))}$.

### Probability distributions

A **Bernoulli distribution** can have the values 0 (failure) and 1 (success). It takes the probability of success $p$ as a parameter. $q$ is the probability of failure and is calculated $q=1-p$.  
$E[X]=p$  
$Var(X)=pq$

A **Geometric distribution** is the number of Bernoulli trials that before the first success (can be inclusive or exclusive of the success trial). It only takes the probability of success $p$.  
${PDF}_{Geom}(x, p)=(1-p)^xp$  
$E[X]=\frac{q}{p}$  
$Var(X)=\frac{q}{p^2}$

A **Binomial distribution** is the number of successes in $n$ Bernoulli trials. Therefore it takes both the probability of success $p$ and the number of trials $n$.  
${PDF}_{Bin}(x,n,p)=\frac{n!}{x!(n-x)!}p^x(1-p)^{n-p}$  
$E[X]=np$  
$Var(X)=npq$

A **Poisson distribution** is used to model the number of rare events that occur in a period of time. It takes the event rate $\lambda$.  
${PDF}_{Poiss}(x,\lambda)=e^{-\lambda}{\large\frac{\lambda^x}{x!}}$  
$E[X]=\lambda$  
$Var(X)=\lambda$

A **Uniform distribution** represents a model where all values are equally likely. It only takes the number of possibilities $n$.  
${PDF}_{U}(n)=1/n$

A **Normal distribution** or a **Gaussian distribution** is Normally distributed along a bell curve with mean $\mu$ and variance $\sigma^2$.  
${PDF}_(x,\mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}}exp\{-\frac{1}{2\sigma^2}(x-\mu)^2\}$


An **Exponential distribution** models the time between rare events and therefore only accepts non-negative values. It takes the event rate $\lambda$.  
${PDF}_{Exp}(x, \lambda)=\lambda e^{-\lambda x}$  
$E[X]=\frac{1}{\lambda}$  
$Var(X)=\frac{1}{\lambda^2}$

### Inference

The **likelihood** of observing a set of data $D$ is $P(D|\theta)$ where $\theta$ is the set of parameters for our model. This can be calculated as $P(D|theta)=\underset{i}\prod{P(D_i|\theta)}$.

The **prior distribution** is the distribution before making any observations, denoted $P(\theta)$.

The **posterior distribution** is the distribution estimated from the observed data, denoted $P(\theta|D)$. It can be calculated using $P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$ where $P(D)$ is the normalisation constant.

The **normalisation constant** is denoted $P(D)$.

### Simulation

The **inversion method** produces a random number based on the $F$ cumulative distribution using the uniform distribution $U$. The random number $X$ is generated $X=F^{-1}(U(0,1))$.

The **poisson process** counts the number of events in the time interval $[0,t]$.

### Markov chains
The **Markov property** refers to the property where the next state $\pi_{i+1}$ in a random walk depends only on the current state $\pi_{i}$.
$P(X_{n+1}|X_{n}=x_{n}, \ldots,X_{1}=x_{1})=P(X_{n+1}|X_{n}=x_{n})$

The **transition matrix** contains the probabilities of traveling from state to state, usually denoted $a_{pn}$ where $p$ is the previous state and $n$ is the next state.

A **realisation** is a possible random walk of the Markov chain.

### Hidden Markov Models

A **hidden markov model (HMM)** is just a Markov model with an extra symbol sequence where the probability of each symbol $x$ emitted is dependent on the current state $\pi$.

The **emission matrix** contains the probabilities of each state emitting each specific symbol, usually denoted $e_k(x)$ where $l$ is the state and $l$ is the emitted symbol.

The **Viterbi algorithm** is a dynamic programming algorithm for finding the most likely sequence of hidden states from a sequence of symbols. It involves building a matrix of maximum scores (the maximum path score to that node) and a matrix of the previous state that made the maximum path score, then backtracking using the previous state matrix from the node with the highest maximum score results in the most likely sequence. $Sc_{i\ j}=P_{symbol}(x_j|S_j)\times \underset{r}max(Sc_{r\ j-1}\times P_{transition}(Sc_{i\ j}|Sc_{r\ j-1}))$ where $Sc$ is the score matrix, $P_{symbol}(A|B)$ is probability of the symbol $A$ given the state $B$, $x$ is the inputted symbol sequence.

The **forward algorithm** builds the forward matrix and then sums the last column to find the probability of observing a symbol sequence regardless of any particular state path, $P(x)$.

The **forward matrix** can be calculated using $f_{k}(i)=e(\pi_{k},x_i)\sum_{l=1}{f(l,i-1)a(\pi_l,\pi_k)}$.

The **backward algorithm** is very similar to the forward algorithm in the opposite direction in that it builds a backward matrix and then sums the probabilities of the first column for $P(x)$ i.e. $P(x)=\sum_{l=1}a_{0l}e_{l}(x_{1})b_{l}(1)$.

The **backward matrix** is calculated using $b_{k}(i)=\sum_{l=1}{a_{kl}e_{l}(x_{i+1})b_{l}(i+1)}$.

Using both the forward and backwards algorithms, you can calculate $P(\pi_{i}=k|x)=\large\frac{f_k(i)b_k(i)}{P(x)}$.

**Empirical counts** can be used to estimate the probabilities of emissions $\hat{a}_{kl}=\large\frac{A_{kl}}{\sum_i{A_{ki}}}$ and transitions $\hat{e}_{k}(b)=\large\frac{E_{k}(b)}{\sum_j{E_{k}(j)}}$ where $A_{kl}$ and $E_k(b)$ are the counts of occurences in the given state sequence and symbol sequence in addtion to any pseudo-counts.

A **pseudo-count** is a small number to add to the empirical counts so that none are zero.


The **Baum-Welch** algorithm can be used to estimate the parameters $a$ and $e$ of an HMM given a training set of sequences by iteratively approximating the empirical counts $A$ and $E$ and stopping when the latest iteration has only a very small change.  

## Alignment and phylogenetics

### Genetics

**Deoxyribonucleic acid (DNA)** is one of the symbol sets containing genetic information. $\text{DNA}=\{A,C,G,T\}$.

**Ribonucleic acid (RNA)** or **Messenger ribonucleic acid (mRNA)** is another one of the symbol sets containing genetic information. $\text{RNA}=\{A,C,G,U\}$.

The **purines** are $A$ and $G$.  
The **pyramidines** are $C$, $T$ and $U$.

**Proteins** are the twenty amino acids represented by the alphabet: $\{A,R,N,D,C,E,Q,G,H,I,L,K,M,F,P,S,T,W,Y,V\}=\text{Alphabet}\backslash\{B,J,O,U,X,Z\}$.

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

### Alignment
Two sequence regions are **homologous** that share a common ancestry. The level of similarity depends on how recently they shared a common ancestor.

**Orthology** occurs when two genes are separated by a speciation event and evolve independently from there on.  
**Paralogy** occurs when a region of the genome is duplicated in the same genome (a duplication event) and they evolve in parallel in the same genome. The two copies are said to be paralogs.

**Pairwise alignment** is the problem of optimally aligning two sequences.

The probability of getting sequence $x$ is $P(x)=\prod_{i=1}^n\large q_{x_i}$ where $q_{a}$ is the probability of getting state $a$ and $n$ is the length of the sequence.

The **joint likelihood** of an alignment is $P(x,y)=\prod_{i=1}^n\large q_{x_i}q_{y_i}$.

Assuming two sequences are related, the probability of getting $a$ and $b$ is $p_{ab}$.  
The **score** of an alignment is the sum over the local score $s(a,b)$, where $s(a,b)=log(\large\frac{p_{ab}}{q_aq_b})$.

A **linear gap penalty** is defined as $\gamma(k)=-dk$ where $d>0$ and $k$ is the gap length.  
An **affine gap penalty** is defined as $\gamma(k)=-d-(k-1)e$ where $d>e>0$ and $k$ is the gap length. $d$ is the gap open penalty and $e$ is the gap extension penalty.

**Global alignment** refers to the optimal alignment of whole sequences.  
The **Needleman-Wunsch algorithm** can be used to find a global alignment.

**Local alignment** refers to the optimal alignment of some subsequences within each sequence.  
The **Smith-Waterman algorithm** can be used to find a local alignment.

### Multiple sequence alignment

**Multiple sequence alignment (MSA)** is the alignment of more than two sequences.

**Progressive alignment** is a widely-used heuristic technique for finding a good enough multiple sequence alignment. It involves iteratively using pairwise alignment on two sequences and using the aligned sequences with others.

**Unweighted pair group method using arithmetic averages (UPGMA)** is an $O(n^2)$ method to decide the ordering of pairwise alignments in a multiple sequence alignment.  
The distance between two clusters is the average distance between all pairs between clusters: $d_{ij}=\frac{1}{|C_i||C_j|} \sum_{x\in C_i,y\in C_j} d_{xy}$ where $|C|$ is the number of sequences in the cluster and $d_{xy}$ is a distance function/matrix between two clusters.

The **Feng-Doolittle algorithm** uses a UPGMA tree and aligns sequences in the order they were added to the tree. A MSA is aligned with another MSA based on the two sequence elements (one from each MSA) that gives the highest scoring alignment.  
Any gap characters can be replaced with a neutral character $X$, which can be aligned to any other character (gap or residue) at no cost.

### Parsimony

The **maximum parsimony tree** is the tree that requires the fewest changes along it to explain all sequences.

The **parsimony algorithm** involves finding the set of possible ancestor bases at each site and when an empty set is found, the cost is incremented.

The **weighted parsimony algorithm** uses a separate score cost for each base to be the ancestor root at that site.

**Branch and bound** algorithms involve abandoning possibilities once they are of the same or worse score than the current best possibility. This is based on the idea that the score will only either stay the same or get worse with more additions.

**Heuristic search** gives up on finding the maximum parsimony tree, and attempts to find a good-enough tree in a much more efficient manner.

**Subtree prune regraft (SPR)** involves choosing random edges and reattaching them at other random edges to reconstruct the tree.

**Probems with parsimony**:
1. Ignores hidden mutations, and therefore underestimates branch lengths.
2. The more data on mutations can result in higher certainty in the wrong topology, because the randomness of mutations can be mistaken for homology.
