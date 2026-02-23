# Project Development Plan for Anonymised Facial Expression Recognition

## Task framing and success criteria

This project is a privacy–utility co-optimization problem in computer vision: transform an input face image into an anonymized face image such that identity is hard to recover for a strong face recognizer (privacy), while the facial expression signal remains usable for expression recognition (utility). This framing corresponds closely to the modern face de-identification literature, which evaluates anonymization by both privacy protection and preservation of downstream utility. citeturn1search18turn0search21turn0search4

A practical, code-testable success definition should be expressed as a **frontier** rather than a single scalar score: for each anonymization setting (model choice + hyperparameters), measure (a) identity leakage under a strong attacker and (b) expression retention/accuracy, then plot the privacy–utility trade-off curve and identify reliable “knee points” (high privacy gain for minimal utility loss). This “frontier-first” evaluation is consistent with how GAN-based anonymizers like **entity["company","GitHub","code hosting platform"]-hosted** DeepPrivacy/DeepPrivacy2, GANonymization, and CIAGAN motivate control and measurement of the anonymization strength versus usefulness. citeturn8search2turn3search2turn7view0turn8search3turn2search3

To align with the evaluation criteria you provided, define success via four buckets:

**Understanding of the task (purpose):** demonstrate that anonymization must be verified against *strong* recognizers, not weak baselines, and that expression recognition must remain meaningful (not a classifier collapsing to frequent classes). Deep FER surveys explicitly list identity bias and non-expression variations as major challenges in expression recognition, reinforcing why naive anonymization can degrade utility. citeturn6search11

**Methodology (tools, models, real + synthetic training):** use off-the-shelf / pre-trained components (detectors, recognizers, parsers, anonymizers) and focus the “research contribution” on *how to assemble and tune* them: conditioning signals, loss design, robust evaluation, and synthetic-data-driven domain adaptation. GANonymization is explicitly designed to preserve emotional expressions under anonymization, while DeepPrivacy/DeepPrivacy2 provides a strong realistic anonymization baseline and engineering-ready pipeline. citeturn7view0turn7view1turn0search21turn3search2

**Implementation and comparison:** compare multiple anonymizers (classical obfuscation; k-Same-style averaging; modern GAN anonymizers including DeepPrivacy2, CIAGAN, GANonymization) using the same preprocessing, identity attacker, expression evaluator, and metrics. CIAGAN is a CVPR anonymization method with public code and pre-trained weights, which makes it suitable as a strong comparator without training from scratch. citeturn8search3turn7view2turn8search0

**Efforts and contributions (synthetic data + analysis):** show measurable iteration: generate synthetic anonymized datasets (and optionally geometry-based synthetic expressions), retrain or adapt the expression model, and quantify what actually improves the frontier. Clean, reproducible metric computation (e.g., Clean-FID for FID) helps ensure your analysis is credible. citeturn3search3turn2search19turn2search3

## Technical problem decomposition and blockers

The key to “CVPR-level narrowness” is to define a small number of *sharp blockers* and solve them with controlled experiments. Below are the blockers that most often stop projects like this from producing interpretable results, plus methods (and corresponding open-source building blocks) that directly address them.

### Blocker: defining privacy under a realistic attacker

**Problem definition:** “Identity is removed” must be operationalized as “a strong face recognizer cannot reliably match the anonymized face to the original identity.” ArcFace-style embeddings are widely used because they are explicitly optimized for discriminative identity features via an additive angular margin, making them a high-bar proxy for identity leakage. citeturn0search18turn0search2turn0search14

**Engineering blocker:** If you only test a weak classifier, you can get false confidence. Worse, you can overfit the anonymizer to a single attacker and still leak identity to other recognizers. The reversibility literature shows that many anonymization methods can be at least partially reversed under strong learning-based attackers, highlighting the need for careful attacker choice and evaluation methodology. citeturn8search1turn8search24

**Method to solve (research + code):**
- Use **ArcFace embeddings** (via InsightFace) as the default attacker feature space. citeturn0search14turn0search18  
- Add an **adaptive attacker** protocol: train a lightweight classifier on *anonymized* outputs (or fine-tune a head on anonymized embeddings) and retest—this tests whether privacy “survives distribution shift.” This aligns with common warnings in anonymization evaluation that “privacy” must consider adaptive adversaries. citeturn8search1turn1search18  
- Optionally add an **attacker ensemble** (e.g., ArcFace + another open model) only after your single-attacker pipeline is stable; otherwise you dilute debugging focus.

### Blocker: expression retention is entangled with identity

Identity and expression overlap in pixels and representations; preserving subtle mouth/eye/eyebrow cues while altering identity-bearing texture is the core scientific challenge. GANonymization directly targets this by synthesizing anonymized faces from a high-level representation with stated expression-preserving goals and downloadable pre-trained models. citeturn0search4turn7view0turn3search18

**Method to solve (research + code):**
- Constrain the anonymizer with an **expression teacher**: penalize divergence between expression predictions on original vs anonymized images (logits or intermediate features). Knowledge distillation formalizes training a student to match a teacher’s soft predictions, which is a clean way to enforce “task invariance” without requiring perfect labels on anonymized images. citeturn4search3turn4search39  
- Use a teacher that is *stronger than the FER-2013 baseline*, ideally pre-trained on in-the-wild expression data (e.g., AffectNet) or a public pre-trained model repo (e.g., EmoNet). AffectNet is explicitly motivated by the limitations of small in-the-wild datasets and is large-scale; EmoNet-style repositories provide pre-trained models for categorical emotions and valence/arousal. citeturn6search0turn6search6turn6search4

### Blocker: conditioning signals that are “privacy-safe” but still preserve expression

The decisive design choice is **what the anonymizer is allowed to see**.

- **Landmarks/pose**: widely used because they are low-dimensional and capture geometry; they support expression cues (mouth opening, brow shape) while reducing texture identity leakage. MTCNN provides detection with landmark estimation; RetinaFace also predicts landmarks and is available via InsightFace. citeturn1search0turn1search12turn1search1turn0search14  
- **Face parsing / semantic masks**: preserve parts and boundaries explicitly (mouth, eyes, eyebrows) and enable losses that focus edits on identity-bearing regions. CelebAMask-HQ provides 19-class masks and public face-parsing code (BiSeNet variants). citeturn4search0turn4search1  
- **3D geometry**: lets you preserve expression/pose while swapping identity-related texture/shape components more explicitly. DECA reconstructs an animatable 3D head from a single image and can generate expression/pose variations; FLAME is a parametric head model (identity/pose/expression) used across 3D face research, with open resources. citeturn5search0turn5search24turn5search9turn5search29

**Engineering blocker:** It is easy to leak identity if your condition contains texture or if your pipeline accidentally passes original pixels into the generator.

**Method to solve:**
- Start with a **single conditioning family** (recommended: landmarks + optional parsing) and freeze it as a “contract,” then iterate.  
- If you adopt inpainting-based anonymization, note DeepPrivacy2 argues inpainting-based anonymization can provide stronger privacy guarantees because the generator does not observe the original privacy-sensitive region directly. citeturn3search2turn7view1

### Blocker: dataset mismatch and preprocessing dominates results

FER-2013 is 48×48 grayscale faces and is known for being low-resolution; Pins Face Recognition is a cropped celebrity face dataset (105 identities, 17,534 faces) collected from entity["company","Pinterest","social media platform"] and distributed on entity["company","Kaggle","machine learning platform"]; CelebA is a large-scale celebrity face dataset with substantial variation and rich attributes/landmarks, and CelebA-HQ is a widely used high-quality subset/derivation in high-resolution generative modeling. citeturn1search3turn1search7turn2search2turn2search28turn2search9turn2search17

**Engineering blocker:** if you do not enforce a consistent “face-crop contract” (alignment, resolution, channels, normalization), you will end up measuring preprocessing artifacts rather than anonymization quality.

**Method to solve:**
- Pick one canonical working resolution for the anonymizer (commonly 256×256 for DeepPrivacy2-like pipelines) and implement strict dataset adapters that upsample/convert FER appropriately while preserving its label split. DeepPrivacy2 explicitly reports improved face anonymization quality and higher resolution compared to earlier DeepPrivacy versions, which supports the practicality of 256×256 processing. citeturn7view1turn3search2  
- Maintain separate “evaluation resolutions” for expression and identity models if needed, but always derive them from the same aligned crop.

### Blocker: GAN training instability and misleading “realism” metrics

pix2pix’s conditional GAN template (U-Net generator + PatchGAN discriminator + reconstruction regularizer) is a proven engineering baseline for conditional synthesis, and GANonymization’s training entry points show a pix2pix-style pathway. citeturn0search3turn0search7turn7view0

For realism metrics, Fréchet Inception Distance (FID) is widely used for generative model evaluation; Clean-FID exists to reduce common implementation inconsistencies and make scores more comparable. citeturn2search3turn2search19turn3search3  
For perceptual similarity / identity-irrelevant structure checks, LPIPS is a standard learned perceptual similarity metric with official code. citeturn4search14turn4search6

**Blocker:** optimizing for FID alone can push the anonymizer toward “average face realism” while still harming expression cues, so realism must remain a *diagnostic*, not the target.

## Open-source ecosystem survey for method and code reuse

This section lists the most relevant open-source projects to leverage directly (so you do not train major models from scratch), and how each maps to a concrete module in your codebase.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["DeepPrivacy2 face anonymization example images","GANonymization face anonymization expression preserving examples","CIAGAN face anonymization examples","ArcFace face recognition embedding visualization"],"num_per_query":1}

### Anonymization models to integrate as first-class backends

**GANonymization (expression-preserving anonymization).** The repository emphasizes expression-preserving abilities, provides a documented training/anonymization interface, and offers downloadable pre-trained models (25-epoch “publication version” and 50-epoch “demo version”). This makes it a top candidate for your “no training from scratch” constraint while still allowing fine-tuning/ablation. citeturn7view0turn0search4

**DeepPrivacy / DeepPrivacy2 (realistic anonymization via GAN; inpainting pipeline).** DeepPrivacy proposes anonymization by generating faces from “privacy-safe information” (pose/background) and explicitly claims the generator avoids observing privacy-sensitive information; DeepPrivacy2 is a refactored toolbox with improved face anonymization quality and higher resolution, and includes a full anonymization pipeline structure in code. citeturn8search2turn8search14turn7view1turn3search2

**CIAGAN (CVPR anonymization baseline).** CIAGAN is a CVPR anonymization method using conditional GANs for controllable de-identification and reports retaining utility for downstream tasks; the official repo includes training and inference scripts and links to pre-trained models. It is highly suitable as a “strong comparator” in your benchmark suite. citeturn8search3turn8search0turn7view2

**Why integrate multiple anonymizers early:** the reversibility/robustness literature shows anonymizers can behave unexpectedly under different attackers and settings; having multiple backends forces your evaluation harness to be model-agnostic and helps prevent overfitting your entire project to a single method. citeturn8search1turn8search24

### Face detection, alignment, and landmarks

**MTCNN** provides joint detection and landmark alignment via a cascaded multi-task framework; it is a widely used baseline when you need stable landmark-based conditioning. citeturn1search0turn1search12

**RetinaFace** is a strong single-stage detector with landmark supervision and is available through the InsightFace ecosystem; this reduces integration friction if you already use ArcFace/InsightFace for identity attacks. citeturn1search1turn1search5turn0search14

### Identity attacker: ArcFace via InsightFace

ArcFace’s additive angular margin loss is designed for highly discriminative identity embeddings, making it a strong and common choice for privacy evaluation. InsightFace provides implementations and related face detection modules (including RetinaFace) in one place, which is valuable for engineering simplicity and reproducibility. citeturn0search18turn0search14

### Expression models and teachers

If you only train a small CNN on FER-2013, the “teacher constraint” may be weak and distort your anonymizer optimization. FER-2013’s low-resolution grayscale nature and label noise are well-known constraints. citeturn1search3turn1search31

To upgrade the teacher without training from scratch:
- Use an open pre-trained expression model such as **EmoNet** (repo provides pretrained models and supports categorical emotions and valence/arousal). citeturn6search6  
- Optionally use a model trained on larger in-the-wild datasets like AffectNet; AffectNet is described as a large-scale facial affect database addressing limitations of small in-the-wild expression datasets. citeturn6search0turn6search4

### Face parsing / semantic conditioning

CelebAMask-HQ provides 19 semantic facial component classes and is a common base for face parsing; the public face-parsing PyTorch repository provides training scripts and pre-trained models, making it suitable as a “plug-in conditioner” for anonymization. citeturn4search0turn4search1

### Quality and similarity metrics for analysis

FID was introduced as part of TTUR GAN evaluation work and remains widely used to compare generated and real distributions; Clean-FID provides a standardized implementation to reduce metric inconsistency across resizing/quantization differences. citeturn2search3turn2search19turn3search3  
LPIPS provides a learned perceptual similarity measure with public code and is widely used as a structure/perceptual distance diagnostic. citeturn4search14turn4search6

## Source-code architecture and development plan

This section “narrows” the project into clear modules and interfaces that a coding agent can implement directly. The guiding principle is: **treat each major research question as an interchangeable component + experiment configuration**, rather than a tangled notebook project.

### Repository layout and core abstractions

Use a monorepo with a strict separation between: (a) data contracts, (b) model wrappers (mostly pre-trained), (c) anonymizers (multiple backends), and (d) evaluation/experiments.

Recommended top-level structure (names are illustrative):

- `configs/`
  - `datasets/` (paths, splits, crop resolution contract)
  - `models/` (identity attacker, expression teacher, parsing model)
  - `anonymizers/` (GANonymization, DeepPrivacy2, CIAGAN, blur/k-same)
  - `experiments/` (privacy–utility sweeps, adaptive attacker, synthetic training)
- `src/`
  - `data/` (dataset adapters, face-crop pipeline, caching)
  - `preprocess/` (detect→align→crop, landmark/parsing extraction)
  - `models/` (wrappers around ArcFace/InsightFace, expression teacher, parsing)
  - `anonymizers/` (uniform interface + backend implementations)
  - `metrics/` (identity leakage metrics, expression metrics, realism diagnostics)
  - `runner/` (experiment orchestration, reproducibility, logging)
- `scripts/` (CLI entry points)
- `reports/` (auto-generated figures, tables, and JSON summaries)

Define a minimal set of “contracts” that everything else depends on:

**FaceCropContract**
- canonical color space (RGB float32)
- canonical anonymizer resolution (e.g., 256×256)
- alignment definition (five landmarks or affine transform metadata)
- optional segmentation mask and landmark heatmaps
- metadata: dataset name, split, subject/identity label (if available), expression label (if available)

**Anonymizer interface**
- `anonymize(batch: FaceCropBatch, mode: str, params: dict) -> AnonymizedBatch`
- Must support: deterministic runs (seed), returning masks/aux outputs (if available), and batched inference.

**Attacker interface (identity)**
- `embed(images) -> embeddings`
- `score_pair(e1, e2) -> similarity`
- optional `fit_adaptive(train_anonymized) -> attacker_head`

**Utility interface (expression)**
- `predict(images) -> logits/probs`
- `calibration` utilities (ECE / reliability curves) if you include them.

This abstraction makes it easy to do “method research” by swapping modules without rewriting pipelines.

### Backend integration plan (no training from scratch)

Implement anonymizer backends as wrappers around existing projects first, before any fine-tuning:

**GANonymizationBackend**
- Load pre-trained weights (publication/demo) via the repo’s provided model downloads and invoke the repo’s anonymization path.
- Expose the repo’s “pix2pix-style” mode and parameters through your unified interface. citeturn7view0turn0search3

**DeepPrivacy2Backend**
- Use the toolbox interface and its detect→anonymize pipeline.
- Keep a knob for face-only anonymization versus pipeline settings (as supported). citeturn7view1turn3search2

**CIAGANBackend**
- Load the provided pre-trained model and implement inference (the repo includes `test.py` guidance).
- Standardize output scaling and alignment assumptions to match your FaceCropContract. citeturn7view2turn8search0

**Classical baselines**
- Blur/pixelate/blackout: implement as a lightweight backend with a “strength” parameter sweep.
- k-Same-style averaging: implement a deterministic baseline that selects k neighbors (in some feature space) and averages faces; the k-Same concept (select k closest and average) is described in the literature and provides a principled privacy–utility reference point. citeturn1search26turn1search10

### Conditioning extraction modules

Implement conditioning as separate, cached preprocessing steps so they do not become the bottleneck:

- `LandmarkExtractor`: MTCNN or RetinaFace landmarks. citeturn1search0turn1search1  
- `FaceParser`: BiSeNet-based face parsing using CelebAMask-HQ pre-trained models for 19-class masks. citeturn4search0turn4search1  
- Optional `GeometryExtractor`: DECA-based 3D reconstruction and expression/pose parameters for synthetic augmentation and/or conditioning. citeturn5search0turn5search24

Store all extracted artifacts in a versioned cache keyed by:
`(dataset, split, image_id, detector_version, crop_contract_hash)`.

### Evaluation harness: make it the “center of gravity”

The evaluation harness should be runnable on any anonymizer backend and produce a consistent output schema (JSON + figures):

- Identity leakage:
  - closed-set top-1 identification accuracy (Pins identities)
  - verification ROC / TAR@FAR using ArcFace similarity
  - adaptive attacker performance (trained on anonymized outputs) citeturn0search18turn2search2turn8search1
- Expression utility:
  - accuracy on FER-2013 (original vs anonymized)
  - “expression consistency” (teacher predictions before/after anonymization)
  - confusion matrix deltas (to detect collapse) citeturn1search3turn6search11turn4search3
- Realism diagnostics:
  - FID (Clean-FID) between anonymized outputs and reference real faces
  - LPIPS between original and anonymized (interpret carefully; it is not directly “utility”) citeturn2search3turn3search3turn4search14

Output artifacts:
- `results/run_{id}/metrics.json`
- `results/run_{id}/frontier.csv`
- `results/run_{id}/plots/*.png` (privacy–utility curves; ablation charts)

## Experimental methodology and performance iteration loop

This section is the “research process engine”: it describes how to iterate for performance while making defensible engineering trade-offs.

### Baseline ladder and comparison matrix

A robust comparison ladder (from simplest to strongest) prevents overinvesting early in fragile methods:

**Tier A: no-learning baselines**
- blur / pixelation / blackout sweeps (privacy strength knob)  
These anchor the “privacy-max / utility-min” and “utility-max / privacy-min” extremes. DeepPrivacy reports that traditional pixelation/blur/blackout behave differently in downstream performance, illustrating why you need baselines. citeturn8search2

**Tier B: deterministic k-anonymity baseline**
- k-Same-style averaging (k knob), optionally “expression-aware neighbor selection” if time permits  
k-Same’s definition (choose k nearest, average) is described in face de-identification comparisons and is useful as a principled deterministic reference. citeturn1search26turn1search10

**Tier C: pretrained GAN anonymizers (no training from scratch)**
- DeepPrivacy2 pretrained
- CIAGAN pretrained
- GANonymization pretrained  
All three have public code and documented usage; GANonymization also provides explicit pre-trained model downloads and a CLI entry point. citeturn7view1turn7view2turn7view0

### The core research ablations

These ablations map directly to your proposal’s research questions but remain narrow enough for a coding agent to implement cleanly.

**Ablation: privacy-safe conditioning**
- Landmarks only (MTCNN/RetinaFace)
- Parsing masks only (CelebAMask-HQ BiSeNet)
- Landmarks + parsing
- Optional geometry (DECA/FLAME)  
This directly tests which conditioning family best preserves expression while enabling identity suppression. citeturn1search0turn1search1turn4search0turn5search0turn5search9

**Ablation: objective variants for expression preservation**
- Teacher-logit consistency loss (KD-style)
- Feature-level consistency loss (teacher intermediate features)  
Distillation is a well-established way to transfer soft targets; it is a clean tool for “keep the task invariant.” citeturn4search3turn4search39

**Ablation: identity suppression loss**
- cosine similarity penalty in ArcFace embedding space
- threshold-based verification objective (push below match threshold)  
ArcFace is a standard identity embedding and provides a strong base for privacy evaluation. citeturn0search18turn0search14

**Ablation: attacker robustness**
- fixed attacker only
- adaptive attacker trained on anonymized domain
- (optional) attacker ensemble  
Reversibility research motivates not trusting single-attacker results as the end of story. citeturn8search1turn8search24

### Synthetic data generation strategy

Synthetic data is not “extra”; it is the most direct way to improve expression performance under anonymized-domain shift while demonstrating effort and analysis.

**Synthetic data type one: anonymized outputs as synthetic training domain**
- Generate anonymized versions of FER-2013 (and optionally other face datasets) using each anonymizer backend.
- Retrain or fine-tune the expression classifier on:
  - real-only
  - anonymized-only
  - mixed real+anonymized
- Report how the privacy–utility frontier changes (does expression recover without increasing identity leakage?)  
This leverages the fact that GAN anonymizers already create synthetic images; DeepPrivacy2 is explicitly positioned as a toolbox that can anonymize data used for downstream training, supporting this direction. citeturn7view1turn3search2turn1search3

**Synthetic data type two: geometry-driven expression augmentation (optional but high-impact)**
- Use DECA to reconstruct 3D heads and render controlled expression/pose variations; then anonymize these renders to produce large diverse anonymized-expression data.
- Label synthetic renders using the expression teacher (pseudo-labeling) and validate with a held-out real test set.  
DECA is explicitly designed to reconstruct an animatable 3D head from a single image and supports animation with various poses/expressions, making it an engineering-feasible route for controllable synthetic augmentation. citeturn5search0turn5search24

### Performance improvement loop

To keep the “process” central, enforce an iteration cadence:

1. **Stabilize preprocessing and baseline attackers** (identity and expression must be strong on original images first).  
   - ArcFace attacker verified on Pins (closed-set) and verification metrics. citeturn0search18turn2search2  
   - FER-2013 baseline verified, acknowledging low-res grayscale properties. citeturn1search3turn1search31

2. **Run Tier A/B baselines to validate the harness** (you should see obvious privacy–utility behavior).

3. **Integrate pretrained anonymizers (Tier C)** and produce initial frontiers.

4. **Pick one “mainline” model (recommended: GANonymization)** and apply controlled fine-tuning only after the harness is trustworthy. GANonymization explicitly targets expression preservation and provides pretrained weights, which is ideal under your constraints. citeturn7view0turn0search4

5. **Use synthetic anonymized training** to recover expression performance, then verify identity leakage did not degrade under adaptive attack.

## Deliverables, evaluation mapping, and documentation

The final deliverable should look like a reproducible benchmark + an “improvement narrative,” not just a single model checkpoint.

**Deliverable: reproducible codebase**
- one-command scripts for:
  - preprocessing and caching face crops
  - running each anonymizer backend
  - training/fine-tuning expression models (starting from pretrained)
  - computing privacy–utility frontiers and exporting plots

**Deliverable: comparison report with controlled evidence**
- A baseline table/figure set:
  - privacy–utility curves for blur/k-Same/DeepPrivacy2/CIAGAN/GANonymization
  - adaptive attacker results (identity leakage after attacker adapts)
  - confusion matrix shifts for expression (to rule out trivial “consistency”)
  - realism diagnostics (Clean-FID + qualitative grids)

**Deliverable: contribution narrative**
- A clear ablation story showing how each intervention moves the frontier:
  - conditioning choice
  - identity-loss choice
  - expression-teacher constraint choice
  - synthetic anonymized training and its effect

This structure directly matches the rubric:
- **Understanding:** precise threat model + utility definition, grounded in strong recognizers and known FER challenges. citeturn0search18turn6search11turn8search1  
- **Methodology:** justified use of pretrained anonymizers and teacher constraints; conditioning choices supported by established detectors/parsers/3D tools. citeturn7view0turn7view1turn1search0turn4search0turn5search0  
- **Implementation & comparison:** multiple open-source anonymizers + consistent evaluation harness. citeturn7view2turn8search3turn3search3  
- **Effort & contributions:** synthetic anonymized datasets + (optional) DECA-driven augmentation + analysis of measurable frontier shifts. citeturn7view1turn5search0turn2search3