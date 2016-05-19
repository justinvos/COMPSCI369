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

The **Newton-Raphson method** is a faster algorithm for finding a root of an equation. It involves iteratively approximating the root using: \(x_{i+1}=x_i - \frac{f(x_i)}{f'(x_i)}\).

## Genetics

**Deoxyribonucleic acid (DNA)** is one of the symbol sets containing genetic information. \(DNA=\{A,T,G,C\}\)

**Translation** is the process of converting DNA into mRNA.  
**Transcription** is the process of converting mRNA into proteins.

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

## Multiple sequence alignment

**Multiple sequence alignment (MSA)** is the alignment of more than two sequences.

**Progressive alignment** is a widely-used heuristic technique for finding a good enough multiple sequence alignment. It involves iteratively using pairwise alignment on two sequences and using the aligned sequences with others.

**Unweighted pair group method using arithmetic averages (UPGMA)** is a method to decide the ordering of pairwise alignments in a multiple sequence alignment.  
The distance between two clusters is the average distance between all pairs between clusters: \(d_{ij}\frac{1}{|C_i||C_j|} \sum_{x\in C_i,y\in C_j} d_{xy}\) where \(|C|\) is the number of sequences in the cluster and \(d_{xy}\) is a distance function/matrix between two clusters.

## Parsimony

The **maximum parsimony tree** is the tree that requires the fewest changes along it to explain all sequences.
