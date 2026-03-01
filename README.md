# Multi-omics integration by causal network inference using CORNETO

This is a group project based tutorial first presented at the 2026 EMBL-EBI
_Introduction to multi-omics data integration and visualisation_ training.
The fundamental idea of the project is to infer molecular activities (e.g.
transcription factor or kinase activities) from different omics modalities, map
the activities to a prior-knowledge network, and then use optimisation methods
to infer networks that the most plausibly explain the activity patterns given
the known perturbations.
This project carries quite some similarities with the [COSMOS project led by
Aurelien Dugourd](https://github.com/saezlab/COSMOS_basic). However, there are
important differences:
- The COSMOS project uses metabolomics in addition to transcriptomics, while
    here we can use phosphoproteomics, proteomics, and secretomics
- The COSMOS project uses the `cosmosR` R package with its dedicated
    prior-knowledge network: this is an out-of-the-box solution which uses the
    old R implementation of the CARNIVAL network optimisation method
- Within the next 1-2 years the original COSMOS will be replaced by much more
    customisable and feature-rich solutions within the Python ecosystem
- In this project, we use the already available new Python components to
    perform a workflow very similar to COSMOS: we have to do more things "by
    hand", which allows us more freedom and also helps understanding

## Data and study background

Our example data comes from a study of kidney fibrosis by Tüchler *et al.*
(2025). It employs an _in vitro_ model of TGF-beta-induced fibrotic
transformation of PDGFR-beta positive patient derived kidney mesenchymal cells.
These cells are similar to myofibroblasts, the main cell type responsible for
excess extracellular matrix (ECM) deposition in fibrosis. The study generated a
comprehensive time-resolved multi-omics dataset spanning 7 time
points (5 min to 96 h) and 4 omics modalities under 10 ng/l TGF-beta
stimulation, quantifying over 14,000 biomolecules. The omics modalities include
transcriptomics, phosphoproteomics, proteomics, and secretomics, complemented
with imaging of _collagen I_ deposition. Furthermore, it also correlates and
validates its results by comparing to kidney fibrosis patient data, as well as
_in vitro_ data from patient lung fibroblasts and tissue slices. Another
important validation is the quantification of collagen I deposition under
knock-down of transcription factors most significant based on the main
experiment. Importantly, while we also measure collagen I as transcript and
protein, the actual ECM fiber deposition doesn't only depend on collagen I, but
several further proteins which cross-link with it, or build or degrade the ECM.
A general observation in the study is that the early (especially 1h) and late
response to TGF-beta is different. To model the time dynamics, multiple network
inference steps are performed: the initial with only TGF-beta as perturbation,
then the early response where hits in early secretomics are added as additional
inputs, early-late where still using the same eatly inputs the network is
inferred with activities from late omics; and the late response where inputs
are derived from the late secretomics.

## Tools

Our workflow consists of tools developed in the [Saez
Lab](https://saezlab.org/):

- **Decoupler** for activity inference
- **OmniPath** for prior knowledge network retrieval
- **CORNETO** for integer linear (ILP) network optimization

These are the key components of the mechanistic modeling approach used and
developed in our group. To use better the limited time available in the
training, we will focus 90% on the CORNETO part, using only `corneto` (with its
dependencies) and standard data handling and visualisation tools.

**Reference:** Tüchler, Burtscher *et al.* "Dynamic multi-omics and
mechanistic modeling approach uncovers novel mechanisms of kidney fibrosis
progression." *Mol. Syst. Biol.* 21:1030–1065 (2025).

References needed:
Software: OmniPath, Decoupler, CORNETO, CARNIVAL, COSMOS
The kidney patient validation paper used and referenced in Tuchler et al.

## Setup

We will use [`uv`](https://docs.astral.sh/uv/) for an efficient and clean
management of dependencies and our environment. In the training we use Python
3.12 because it's the default version on the virtual machines. The first step
is to clone the project repository where I included all the required data,
documentation, example scripts and results.

```
git clone https://github.com/saezleb/corneto-ebi-multiomics
```

The repo contains the environment definition in `pyproject.toml`, you can set
up the enviromnent with `uv`:

```bash
cd corneto-ebi-multiomics
uv sync
```

Alternatively, install in a virtual environment:

```bash
uv venv
uv pip install -e .
```

From now on, every time you want to run something in this environment, just add
`uv run` in front of the command; and if you want to install more dependencies,
you can use `uv add <package-name>`. For example, to start a Python shell:

```
uv run python
```

## Repository structure

```
data/
  differential/
    diff_expr_all.tsv          # Differential expression, all omics & time points
    enzyme_activities.tsv      # TF and kinase activity scores
  network/
    paper_edges.tsv            # Published network edges (for comparison)
    paper_nodes.tsv            # Published network nodes (for comparison)
  imaging/
    col1_timecourse.tsv        # COL1 imaging data (for interpretation)
scripts/
  01_decoupler_demo.py         # TF activity inference with Decoupler
  02_prepare_inputs.py         # Data preparation and PKN retrieval
  03_corneto_network.py        # CARNIVAL network optimization
  04_visualize_results.py      # Network visualization and interpretation
results/                       # Generated outputs (networks, figures)
```

## Tutorial outline

### Introduction (30 min): Activity inference with Decoupler

We start by demonstrating how transcription factor (TF) activities can be
inferred from transcriptomics data using Decoupler and the CollecTRI
regulon database. This shows how abstract molecular activities can be
estimated from omics measurements, motivating the network integration step.

**Script:** `scripts/01_decoupler_demo.py`

### Session 1 (1.5 h): Data preparation and prior knowledge network

We load the differential omics data, select time points and modalities,
and prepare the inputs for CORNETO. We retrieve a signed, directed
protein interaction network from OmniPath, filter it for expressed genes,
and prune it for reachability.

**Script:** `scripts/02_prepare_inputs.py`

### Session 2 (1.5 h): Network inference with CORNETO

Using the CARNIVAL algorithm implemented in CORNETO, we find an optimal
subnetwork of the prior knowledge network that connects upstream
perturbations (TGF-beta stimulus, kinase and TF activities) to downstream
measurements (secreted protein changes). The optimization balances data
fit against network parsimony.

**Script:** `scripts/03_corneto_network.py`

### Session 3 (1.5 h): Visualization and interpretation

We visualize the inferred network, compare it to the published results,
and interpret the findings in the context of kidney fibrosis biology.

**Script:** `scripts/04_visualize_results.py`

## Data description

The input data comes from the supplementary tables of Tüchler *et al.* (2025):

- **Differential expression** (`diff_expr_all.tsv`): Log fold changes and
  adjusted p-values for all omics modalities (rna, proteomics,
  phosphoproteomics, secretomics) across 7 time points after TGF-beta
  stimulation vs. control. 391,105 measurements.

- **Enzyme activities** (`enzyme_activities.tsv`): Transcription factor and
  kinase activity scores inferred by Decoupler from transcriptomics
  (for TFs, using CollecTRI) and phosphoproteomics (for kinases, using
  a kinase-substrate network). 128 significant activities.

- **Imaging data** (`col1_timecourse.tsv`): Collagen I fluorescence
  intensity from immunofluorescence microscopy across the time course,
  providing a phenotypic readout of fibrotic ECM deposition.

## Key concepts

- **Prior knowledge network (PKN):** A signed, directed graph of known
  protein-protein regulatory interactions from OmniPath. Edges indicate
  activation (+1) or inhibition (-1), making it a causal network suitable for
  mechanistic modeling.

- **CORNETO:** "Core network optimiser" - a Python framework that is able to
    deliver diverse network optimisation problems by diverse formulations to a
    number of optimisation solvers (backends).

- **CARNIVAL:** An optimisation method that finds a subnetwork of the PKN
  consistent with observed perturbations (inputs) and measurements (outputs),
  while penalizing network complexity (L0 regularization on edges). It is one
  of the many optimisation methods CORNETO supports.

- **CarnivalFlow:** The current CORNETO implementation of CARNIVAL, using
  a flow-based formulation that supports multi-sample analysis with
  structured sparsity regularization.
