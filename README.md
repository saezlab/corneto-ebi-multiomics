# Multi-omics integration by causal network inference using CORNETO

This is a group project based tutorial first presented at the 2026 EMBL-EBI
_Introduction to multi-omics data integration and visualisation_ training.
The fundamental idea of the project is to infer molecular activities (e.g.
transcription factor or kinase activities) from different omics modalities, map
the activities to a prior-knowledge network, and then use optimisation methods
to infer networks that the most plausibly explain the activity patterns given
the known perturbations.
This project carries quite some similarities with the [COSMOS project led by
Aurelien Dugourd](https://github.com/saezlab/COSMOS_basic) (Dugourd *et al.*,
2021; 2024). However, there are important differences:
- The COSMOS project uses metabolomics in addition to transcriptomics, while
    here we can use phosphoproteomics, proteomics, and secretomics
- The COSMOS project uses the `cosmosR` R package with its dedicated
    prior-knowledge network: this is an out-of-the-box solution which uses the
    old R implementation of the CARNIVAL (Liu *et al.*, 2019) network
    optimisation method
- Within the next 1-2 years the original COSMOS will be replaced by much more
    customisable and feature-rich solutions within the Python ecosystem
- In this project, we use the already available new Python components to
    perform a workflow very similar to COSMOS: we have to do more things "by
    hand", which allows us more freedom and also helps understanding

## Data and study background

Our example data comes from a study of kidney fibrosis by [Tüchler *et al.*
(2025)](#references). It employs an _in vitro_ model of _TGF-beta_-induced fibrotic
transformation of PDGFR-beta positive patient derived kidney mesenchymal cells.
These cells are similar to myofibroblasts, the main cell type responsible for
excess extracellular matrix (ECM) deposition in fibrosis. The study generated a
comprehensive time-resolved multi-omics dataset spanning 7 time
points (5 min to 96 h) and 4 omics modalities under 10 ng/l _TGF-beta_
stimulation, quantifying over 14,000 biomolecules. The omics modalities include
transcriptomics, phosphoproteomics, proteomics, and secretomics, complemented
with imaging of _collagen I_ deposition. Furthermore, it also correlates and
validates its results by comparing to kidney fibrosis patient data, as well as
_in vitro_ data from patient lung fibroblasts and tissue slices. Another
important validation is the quantification of _collagen I_ deposition under
knock-down of transcription factors most significant based on the main
experiment. Importantly, while we also measure _collagen I_ as transcript and
protein, the actual ECM fiber deposition doesn't only depend on _collagen I_, but
several further proteins which cross-link with it, or build or degrade the ECM.
A general observation in the study is that the early (especially 1h) and late
response to _TGF-beta_ is different. To model the time dynamics, multiple network
inference steps are performed: the initial with _TGF-beta_ as the only perturbation
is responsible for the early TF and kinase activity patterns; then the early
activities (plus _TGF-beta_) become the inputs and the early secretomics hits are
the targets; next the early transcriptomics considered to be the perturbations
and the late activities are the outputs; and finally, the connections between
these late activity hits and the late secretomics are inferred.

## Tools

Our workflow consists of tools developed in the [Saez
Lab](https://saezlab.org/):

- **Decoupler** (Badia-i-Mompel *et al.*, 2022) for activity inference
- **OmniPath** (Türei *et al.*, 2026) for prior knowledge network retrieval
- **CORNETO** (Rodriguez-Mier *et al.*, 2025) for integer linear (ILP)
  network optimization

These are the key components of the mechanistic modeling approach used and
developed in our group. To use better the limited time available in the
training, we will focus 90% on the CORNETO part, using only `corneto` (with its
dependencies) and standard data handling and visualisation tools.

### References

- Tüchler N, Burtscher ML *et al.* "Dynamic multi-omics and mechanistic
  modeling approach uncovers novel mechanisms of kidney fibrosis progression."
  *Mol. Syst. Biol.* 21:1030–1065 (2025).
  [doi:10.1038/s44320-025-00116-2](https://doi.org/10.1038/s44320-025-00116-2)
- Rodriguez-Mier P, Garrido-Rodriguez M, Gabor A, Saez-Rodriguez J. "Unifying
  multi-sample network inference from prior knowledge and omics data with
  CORNETO." *Nat. Mach. Intell.* 7:1168–1186 (2025).
  [doi:10.1038/s42256-025-01069-9](https://doi.org/10.1038/s42256-025-01069-9)
- Türei D, Schaul J, Palacio-Escat N, Bohár B *et al.* "OmniPath: integrated
  knowledgebase for multi-omics analysis." *Nucleic Acids Res.*
  54(D1):D652–D660 (2026).
  [doi:10.1093/nar/gkaf1126](https://doi.org/10.1093/nar/gkaf1126)
- Badia-i-Mompel P *et al.* "decoupleR: ensemble of computational methods to
  infer biological activities from omics data." *Bioinform. Adv.* 2:vbac016
  (2022). [doi:10.1093/bioadv/vbac016](https://doi.org/10.1093/bioadv/vbac016)
- Liu A, Trairatphisan P, Gjerga E *et al.* "From expression footprints to
  causal pathways: contextualizing large signaling networks with CARNIVAL."
  *npj Syst. Biol. Appl.* 5:40 (2019).
  [doi:10.1038/s41540-019-0118-z](https://doi.org/10.1038/s41540-019-0118-z)
- Dugourd A *et al.* "Causal integration of multi-omics data with prior
  knowledge to generate mechanistic hypotheses." *Mol. Syst. Biol.*
  17:e9730 (2021).
  [doi:10.15252/msb.20209730](https://doi.org/10.15252/msb.20209730)
- Dugourd A, Lafrenz P, Mañanes D, Paton V, Fallegger R, Kroger AC, Türei D,
  Shtylla B, Saez-Rodriguez J. "Modeling causal signal propagation in
  multi-omic factor space with COSMOS." *bioRxiv* (2024).
  [doi:10.1101/2024.07.15.603538](https://doi.org/10.1101/2024.07.15.603538)
- Kuppe C *et al.* "Decoding myofibroblast origins in human kidney fibrosis."
  *Nature* 589:281–286 (2021).
  [doi:10.1038/s41586-020-2941-1](https://doi.org/10.1038/s41586-020-2941-1)

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
    diff_expr_all.tsv.gz       # Differential expression, all omics & time points
    activities.tsv             # TF and kinase activity scores
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

## Group project schedule

### Session 1 (1:45 h): Intro and activity inference with Decoupler

We start by demonstrating how transcription factor (TF) activities can be
inferred from transcriptomics data using Decoupler (Badia-i-Mompel *et al.*,
2022) and the CollecTRI regulon database. This shows how abstract molecular
activities can be estimated from omics measurements, motivating the network
integration step.

**Script:** `scripts/01_decoupler_demo.py`

### Session 2 (2:15 h): Data preparation and prior knowledge network

We load the differential omics data, select time points and modalities,
and prepare the inputs for CORNETO. We retrieve a signed, directed
protein interaction network from OmniPath (Türei *et al.*, 2026), filter it
for expressed genes, and prune it for reachability.

**Script:** `scripts/02_prepare_inputs.py`

### Session 3 (2:00 h): Network inference with CORNETO

Using the CARNIVAL algorithm (Liu *et al.*, 2019) implemented in CORNETO
(Rodriguez-Mier *et al.*, 2025), we find an optimal subnetwork of the prior
knowledge network that connects upstream perturbations (_TGF-beta_ stimulus,
kinase and TF activities) to downstream measurements (secreted protein
changes). We can do more than one network inference, depending on time maybe we
start network inference already in the previous session. In this project,
because the longitudinal nature of the data poses additional challenge, we can
infer an early and a late network, following similar steps as in the paper.

**Script:** `scripts/03_corneto_network.py`

### Session 4 (3:30 h): Visualization and interpretation

We visualize the inferred network, the activities, compare them to the
published results, and interpret the findings in the context of kidney fibrosis
biology. We also have available the _collagen I_ deposition data from imaging.

**Script:** `scripts/04_visualize_results.py`

## Data description

The input data comes from the supplementary tables of [Tüchler *et al.*
(2025)](#references):

- **Differential expression** (`diff_expr_all.tsv.gz`): Log fold changes and
  adjusted p-values for all omics modalities (rna, proteomics,
  phosphoproteomics, secretomics) across 7 time points after _TGF-beta_
  stimulation vs. control. 391,105 measurements.

- **Molecular activities** (`activities.tsv`): Transcription factor and
  kinase activity scores inferred by Decoupler from transcriptomics
  (for TFs, using CollecTRI) and phosphoproteomics (for kinases, using
  a kinase-substrate network). 287 TFs and 157 kinases across 7 time points
  (3,108 rows total).

- **Imaging data** (`col1_timecourse.tsv`): Collagen I fluorescence
  intensity from immunofluorescence microscopy across the time course,
  providing a phenotypic readout of fibrotic ECM deposition.

## Key concepts

- **Activity inference:** when the state captured in omics data is in part a
  mechanistic consequence of certain molecular activities and robust
  relationships between activities and omics measurements exist in
  prior-knowledge (these are called signatures or footprints), it is possible
  to estimate the activities from the omics data by simple statistics similar
  to enrichment analysis (Badia-i-Mompel *et al.*, 2022). The simplest concrete
  example is transcription factor (TF) activity inference: if we know the
  target genes of a TF from prior-knowledge (this is the footprint of the TF,
  aka its regulon), and the majority of those target genes show a positive fold
  change in transcriptomics data, we can infer that the TF gets activated in
  the given condition. Read more in the [Decoupler
  manual](https://decoupler.readthedocs.io).

- **Prior knowledge network (PKN):** A signed, directed graph of known
  protein-protein regulatory interactions from OmniPath (Türei *et al.*,
  2026). Edges indicate
  activation (+1) or inhibition (-1), making it a causal network suitable for
  mechanistic modeling.

- **Network optimisation:** a procedure which, given a set of constraints
  (patterns of molecular activities, perturbations) and a causal
  prior-knowledge network, searches for subnetworks which achieve the least
  amount of logical contradiction between the constaints and the network by
  using the lowest number of nodes and edges from the PKN

- **CORNETO:** "Core network optimiser" (Rodriguez-Mier *et al.*, 2025) - a
  Python framework that is able to deliver diverse network optimisation
  problems by diverse formulations to a number of optimisation solvers
  (backends). Read more in the [CORNETO documentation
  ](https://corneto.org/stable/guide/index.html).

- **CARNIVAL:** An optimisation method (Liu *et al.*, 2019) that finds a
  subnetwork of the PKN consistent with observed perturbations (inputs) and
  measurements (outputs), while penalizing network complexity (L0
  regularization on edges). This method is called causal reasoning, and it is
  one of the many optimisation methods CORNETO supports. Read more in the
  [CARNIVAL paper](https://doi.org/10.1038/s41540-019-0118-z).

- **CarnivalFlow:** The current CORNETO implementation of CARNIVAL, using
  a flow-based formulation that supports multi-sample analysis with structured
  sparsity regularization. [Read more
  here](https://corneto.org/stable/guide/signaling/carnival.html#carnivalflow).

## Further materials

- This is a [notebook from another training
  ](https://colab.research.google.com/drive/1OWKpXrZdpKEK9AWopiViHWkAjjdWq_Hy),
  you find a lot of background information and examples in it
