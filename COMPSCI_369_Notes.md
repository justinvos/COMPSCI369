## Introduction to genetics

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
