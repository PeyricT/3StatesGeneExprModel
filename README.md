# Three-State Gene Expression Model Parameterized for Single-Cell Multi-Omics Data

Authors: Thibaut Peyric, Thomas Lepoutre, Anton Crombach and Thomas Guyet
Contact: thibaut.peyric@inria.fr, anton.crombach@inria.fr, thomas.guyet@inria.fr

We present a novel three-state gene expression model designed to elucidate the underlying mechanisms of mRNA transcription and its regulation. 

Our model incorporates gene regulatory processes by explicitly including a transcription factor-bound state, thereby capturing the dynamic interplay between transcription activation and chromatin dynamics. We fit the model to paired single-cell ATAC-seq and single-cell RNA-seq data, as these data give us simultaneous information on a geneâ€™s transcriptional state and its accompanying chromatin state. Working at the pseudo-bulk level, we extract biologically meaningful high-level descriptors from homogeneous cell (sub)populations, such as the mean and variance of gene expression as well as the fraction of accessible chromatin. 

Crucial to the computational feasibility of our approach, these descriptors can be analytically related to our model parameters. Despite the increased complexity needed to capture regulatory processes in our model, it remains sufficiently parsimonious to infer parameters reliably from experimental data. Each parameter has a clear biological interpretation, reflecting properties such as burst frequency, chromatin opening and closing dynamics, and basal or regulated expression. 

Fitting the model to a large collection of genes allows us to analyze the parameters and distinguish so-called gene expression strategies. The model parameters reveal a small number of distinct expression strategies among gene clusters, providing data-driven novel insight into context-dependent regulation of gene expression.


## Dataset

The PBMC single-cell multi-omics dataset was obtained from 10x Genomics (PBMCs from C57BL/6 mice (v1, 150x150), Single Cell Immune Profiling Dataset by Cell Ranger v3.1.0, 10x Genomics, (2019, July 24)). The dataset was preprocessed and clustered with [Scanpy](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html).
