# Study summary: Tüchler, Burtscher *et al.* (2025)

Tüchler N, Burtscher ML *et al.* "Dynamic multi-omics and mechanistic modeling
approach uncovers novel mechanisms of kidney fibrosis progression." *Mol. Syst.
Biol.* 21:1030–1065 (2025).
[doi:10.1038/s44320-025-00116-2](https://doi.org/10.1038/s44320-025-00116-2)

---

Kidney fibrosis is a hallmark of chronic kidney disease, driven by the
excessive deposition of extracellular matrix (ECM) by myofibroblast-like cells.
TGF-beta signaling is a key driver of this fibrotic transformation, yet the
full cascade of molecular events downstream of TGF-beta — spanning from
minutes to days — remains incompletely understood. This study addresses this
gap by generating a comprehensive time-resolved multi-omics dataset using an
_in vitro_ model of TGF-beta-induced fibrotic transformation. The model uses
PDGFR-beta positive patient-derived kidney mesenchymal cells, which closely
resemble the myofibroblasts implicated in kidney fibrosis _in vivo_ (Kuppe
*et al.*, 2021).

The experiment stimulates these cells with 10 ng/ml TGF-beta and profiles
molecular changes at 7 time points (5 min, 1 h, 12 h, 24 h, 48 h, 72 h,
and 96 h) across 4 omics modalities: transcriptomics (~14,800 genes),
proteomics (~7,000 proteins), phosphoproteomics (~17,200 phosphosites), and
secretomics (~2,200 secreted proteins). This is complemented by
immunofluorescence imaging of collagen I deposition, providing a phenotypic
readout of ECM accumulation. In total, the dataset quantifies over 14,000
biomolecules, making it one of the most comprehensive time-resolved multi-omics
studies of fibrotic signaling.

A central finding is that the cellular response to TGF-beta unfolds in
distinct temporal phases. Early responses (within minutes to hours) are
dominated by phosphorylation signaling — particularly the canonical SMAD2/3
pathway — and rapid transcriptional activation of immediate-early genes. By
12–24 h, a transition occurs toward sustained transcriptomic and proteomic
reprogramming, with upregulation of ECM components, collagens, and
matrix-modifying enzymes. Late responses (48–96 h) involve the actual
secretion and deposition of ECM fibers, which depends not only on collagen I
itself, but on a network of cross-linking, assembly, and degradation factors.

To connect the early signaling events to the late phenotypic outcomes, the
study employs mechanistic network modeling using CORNETO (Rodriguez-Mier
*et al.*, 2025) with the CARNIVAL algorithm (Liu *et al.*, 2019). Molecular
activities of transcription factors and kinases are first inferred from
transcriptomics and phosphoproteomics data using Decoupler (Badia-i-Mompel
*et al.*, 2022). These activities, together with secretome fold changes, are
then mapped onto a prior knowledge network of signed, directed
protein–protein interactions from OmniPath (Türei *et al.*, 2026). The
optimization finds subnetworks that most parsimoniously explain the observed
activity patterns given TGF-beta as the known perturbation. Multiple network
inference steps are performed to capture the temporal dynamics:

| Model | Perturbations (input) | Measurements (output) |
|---|---|---|
| 1. Initial | TGFB1 | early TF/kinase activities |
| 2. Early | early TF/kinase activities + TGFB1 | early secretome |
| 3. Early→late | early secretome + TGFB1 | late TF/kinase activities |
| 4. Late | late TF/kinase activities | late secretome |

In model 1, only TGF-beta is the known perturbation, and the network must
explain the observed early enzyme activities. In model 2, the early activities
together with TGFB1 are taken as given, and the network explains the secretome
changes — this is the model we implement in our tutorial. Models 3 and 4
extend this to the late time points: the early secretome feeds back as
signaling input (via receptors), and late activities or late secretome serve as
measurements, respectively.

The study validates its findings in several ways: by comparing to kidney
fibrosis patient transcriptomics data, by testing the model in lung
fibroblasts and precision-cut tissue slices, and by siRNA knockdown of key
transcription factors identified by the network analysis. This knockdown
experiment confirms that the predicted transcription factors indeed regulate
collagen I deposition, providing functional validation of the computational
network inference approach.
