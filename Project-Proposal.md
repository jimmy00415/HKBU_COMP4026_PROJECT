Expression-Preserving Face Anonymization Under Strong Identity Attackers
Technical Research Project Proposal (CVPR-style, evidence-backed; IEEE citations)
________________________________________
Abstract
Face anonymization is increasingly deployed to reduce identity leakage when sharing or processing visual data. However, many anonymization methods degrade downstream facial expression recognition (FER) performance because identity- and expression-bearing cues are partially entangled in facial appearance and learned representations. This proposal defines a threat-model-aligned, empirically testable objective: learn an anonymization transform that minimizes identity re-identification under strong and adaptive face recognition attackers while preserving expression predictability for FER tasks. We propose a conditional generative anonymization framework grounded in conditional adversarial image-to-image translation [6], identity embedding suppression using a state-of-the-art recognizer proxy (ArcFace) [7], and expression preservation via teacher-guided distillation [12] trained on FER datasets [10], [11]. Evaluation emphasizes privacy–utility trade-offs and includes baselines from classical k-anonymity de-identification [3], utility-aware variants [14], and modern GAN anonymization systems [1], [2], with both fixed and adaptive attacker protocols motivated by the face de-identification literature [5].
Logic bridge: The remainder of this proposal formalizes the problem (privacy + utility), translates the threat model into concrete requirements, then derives the model design and evaluation methodology from those requirements.
________________________________________
1. Background and Motivation
Face data is widely used for analytics, human–computer interaction, and affective computing. When such data is shared or processed beyond a trusted boundary, identity becomes a high-risk attribute: modern face recognition systems achieve extremely strong embedding separability [7], [15], making ad hoc redaction insufficient and motivating principled anonymization approaches [3], [5].
At the same time, many applications require non-identity attributes such as expression (e.g., emotion-aware tutoring, driver monitoring, customer experience analytics). The face de-identification literature repeatedly highlights the privacy–utility tension: strong privacy often correlates with weaker utility for downstream tasks [5], and averaging-based methods can visibly attenuate expression cues [3], [14]. Recent generative anonymization approaches show that conditional synthesis can maintain photorealism and downstream usability better than blunt obfuscation [1], [2], but expression-preserving anonymization is not yet consistently optimized and evaluated under modern attacker assumptions.
Logic bridge: This motivates a formal definition of anonymization as a multi-objective optimization problem with an explicit attacker model and measurable success criteria.
________________________________________
2. Problem Definition
2.1 Formal objective
Given an input face image x, produce an anonymized image x^'=A(x)such that:
	Privacy requirement (identity suppression): A strong face recognizer (attacker) should not reliably match x^'to the identity of x. We will operationalize this using embedding-based and recognition-based metrics with modern recognition backbones (e.g., ArcFace/FaceNet) [7], [15].
	Utility requirement (expression preservation): FER performance on anonymized outputs should remain high, and the predicted expression distribution should remain consistent with the original image under a calibrated FER model trained using robust labeling strategies (FER-2013 / FER+) [10], [11].
This is a constrained optimization (or Pareto) problem: maximize expression utility subject to identity leakage constraints, or trace the privacy–utility frontier.
2.2 Threat model (attacker)
We assume:
	Strong baseline attacker: An off-the-shelf, state-of-the-art recognizer that outputs identity embeddings with high discriminative power (ArcFace, FaceNet) [7], [15].
	Adaptive attacker: The attacker can fine-tune or train a recognizer using anonymized outputs to partially undo distribution shifts introduced by anonymization. This adaptive evaluation is emphasized in face de-identification surveys as necessary for credible privacy claims [5].
	Access assumptions: The attacker sees anonymized images x^'; may know the anonymization algorithm class but not necessarily its random seed or private parameters (Kerckhoffs-style analysis).
Logic bridge: With the threat model fixed, we can now specify concrete engineering requirements and derive a solution that explicitly targets those requirements.
________________________________________
3. Requirements and Success Criteria
3.1 Privacy success criteria
We will report privacy using multiple, attacker-aligned metrics:
	Closed-set identification / verification degradation: accuracy (or TAR@FAR) drop from xto x^'using strong recognizers [7], [15].
	Embedding similarity suppression: reduced cosine similarity between f_"id"  (x)and f_"id"  (x^'), where f_"id" is an ArcFace/FaceNet embedding extractor [7], [15].
	Adaptive attacker robustness: privacy metrics must remain degraded even after attacker adaptation on anonymized data (aligned with modern de-identification evaluation expectations) [5].
3.2 Utility success criteria (FER)
	FER performance retention: accuracy and macro-F1 on FER-2013 / FER+ labels [10], [11].
	Distributional consistency: similarity between expression posterior distributions p(y∣x)and p(y∣x^')to detect “utility collapse” (e.g., outputting a dominant class). FER+ specifically motivates using label distributions to handle noise [11].
3.3 Visual and distributional plausibility
Because trivial corruption can artificially increase “privacy,” we also require:
	Photorealism / distribution alignment consistent with generative anonymization goals demonstrated in prior work [1], [2].
	No obvious artifacts that confound downstream learning, consistent with the objective of retaining usability [1].
Logic bridge: These requirements imply an anonymizer that (i) can modify identity-bearing texture/shape while (ii) preserving expression-relevant deformation and (iii) maintaining realism—suggesting a conditional generative model with explicit identity and expression constraints.
________________________________________
4. Related Work and Gap Analysis
4.1 Classical de-identification and k-anonymity
The k-Same framework provides a formal k-anonymity guarantee for face de-identification via averaging of similar faces, showing ad hoc methods can fail either privacy or utility depending on strength [3]. Utility-aware extensions (e.g., k-Same-Select) incorporate attributes such as expression into the selection process to preserve utility better [14]. While conceptually clean, these methods are limited by averaging artifacts and often assume controlled settings that do not reflect modern open-world recognition pipelines.
4.2 Generative anonymization
Conditional GAN-based anonymization improves realism and background consistency by synthesizing faces from privacy-safe conditionals [1]. CIAGAN extends conditional anonymization with identity-guided generation for images/videos, emphasizing practical anonymization with high visual fidelity [2]. These works motivate generative anonymization as a strong baseline class, but they do not consistently formalize and optimize expression preservation as a primary downstream utility under strong/adaptive recognition attackers.
4.3 Attacker strength in modern face recognition
Face embedding models such as FaceNet and ArcFace define powerful attacker proxies because embedding distances correlate strongly with identity similarity, achieving high verification accuracy and strong separation properties [7], [15]. Privacy evaluation that does not account for such attackers risks overstating protection.
4.4 Expression datasets and label noise
FER-2013 is a widely used benchmark introduced in the “Challenges in Representation Learning” context [10]. FER+ demonstrates that crowd-sourced label distributions can improve training under noisy labels and supports distribution-aware learning objectives [11]. This motivates expression preservation objectives beyond hard-label accuracy alone.
Gap: Existing anonymization pipelines do not consistently provide (i) explicit expression-preserving objectives grounded in FER+ label-distribution insights, and (ii) systematic evaluation under strong + adaptive attackers aligned with modern face recognition capabilities and de-identification survey recommendations [5].
Logic bridge: The proposed method directly targets this gap by coupling identity embedding suppression with expression distillation constraints in a conditional generative framework.
________________________________________
5. Proposed Method
5.1 System overview
We propose a pipeline with three train-time components and one deployment-time component:
Deployment-time:
	Anonymizer A: outputs x^'=A(x).
Train-time supervision and evaluation:
	Identity model f_"id" : a frozen strong recognizer proxy (ArcFace / FaceNet) used to measure and penalize identity leakage [7], [15].
	Expression teacher f_"expr" : a FER model trained on FER-2013/FER+ to produce stable expression posteriors [10], [11].
	Discriminator D: enforces realism in generated outputs, as standard in conditional GAN training [6].
5.2 Preprocessing contract (critical for validity)
Face detection/alignment strongly influences both recognition and FER. We standardize preprocessing using robust detectors/aligners such as MTCNN and/or RetinaFace [8], [9]. A fixed “crop contract” (alignment landmarks, image size, color handling) is enforced across datasets to prevent confounds.
5.3 Model architecture: conditional image-to-image anonymizer
We model anonymization as conditional image-to-image translation using a pix2pix-style generator and PatchGAN discriminator [6], since it is a strong, well-understood baseline for conditional synthesis.
Conditioning strategy: Use privacy-safe or low-identity-leak conditionals (e.g., landmark heatmaps, segmentation/parsing maps, pose descriptors) as in the spirit of privacy-safe conditional generation [1]. The design goal is: preserve expression-relevant geometry while discouraging identity-specific texture replication.
5.4 Training objective derived from requirements
We optimize:
L=λ_"adv"  L_"adv" +λ_"rec"  L_"rec" +λ_"id"  L_"id" +λ_"expr"  L_"expr" .

	Realism (required to avoid trivial privacy): L_"adv" is the conditional adversarial loss [6].
	Structural stability: L_"rec" is a reconstruction / perceptual-structure term to preserve pose and coarse geometry (without enforcing identity texture copying).
	Identity suppression (privacy requirement):
L_"id" minimizes similarity between identity embeddings f_"id"  (x)and f_"id"  (x^')under a strong recognizer proxy [7], [15].
	Expression preservation (utility requirement):
L_"expr" matches the teacher’s soft expression distribution on xand x^'(distillation-style), leveraging the rationale of knowledge distillation [12] and FER+ distribution supervision [11].
Why this mapping is logically necessary:
	The threat model defines the attacker as an embedding-based recognizer; therefore the privacy term must be aligned with embedding leakage, motivating L_"id" using ArcFace/FaceNet [7], [15].
	The utility goal is probabilistic expression consistency under noisy labels; therefore the expression term uses soft distributions (distillation), motivated by [11], [12].
	Realism constraints are needed to prevent trivial privacy gains via corruption and to remain consistent with generative anonymization benefits shown in [1], [2], [6].
5.5 Privacy–utility frontier analysis
We will sweep (λ_"id" ⓜ,λ_"expr"  )to obtain a Pareto frontier. This produces an explicit, quantitative view of trade-offs rather than a single operating point.
Logic bridge: With the method defined, the next section specifies an evaluation protocol that is directly aligned with the stated threat model and success criteria.
________________________________________
6. Experimental Design
6.1 Datasets
	FER-2013: baseline FER benchmark introduced via representation learning challenges [10].
	FER+: relabeled FER with crowd-sourced label distributions to address noise and enable distribution-aware training [11].
	Face imagery for anonymizer realism/capacity: a large-scale face dataset for training stable synthesis (e.g., CelebA-style; dataset choice will follow licensing constraints).
	Identity evaluation set: identities for closed-set recognition tests and adaptive attacker training (dataset selection must ensure consent/licensing; choice does not alter the evaluation methodology).
6.2 Baselines (privacy–utility spectrum)
	Pixel-level obfuscation: blur / mosaic / masking (expected strong privacy but low FER utility).
	k-anonymity de-identification: k-Same averaging [3]; utility-aware k-Same-Select where labels are available [14].
	Modern generative anonymization: DeepPrivacy-style privacy-safe conditional generation [1]; CIAGAN-style conditional anonymization [2].
	Proposed method: conditional GAN + identity embedding suppression + expression distillation.
6.3 Attacker protocols
	Fixed attacker: evaluate identity leakage using ArcFace/FaceNet embeddings without retraining [7], [15].
	Adaptive attacker: train/fine-tune a recognizer on anonymized outputs to test robustness against distribution adaptation, as encouraged by de-identification evaluation guidance [5].
6.4 Metrics
Privacy:
	Verification/identification performance using strong recognizers [7], [15].
	Embedding similarity distribution shift (cosine similarity).
	Adaptive attacker recovery gap (privacy after adaptation vs before).
Utility (expression):
	FER accuracy + macro-F1 on FER-2013/FER+ [10], [11].
	KL divergence / cross-entropy between teacher posteriors on xvs x^'(consistency under distribution supervision) [11], [12].
Realism / non-triviality:
	Distributional quality proxies (e.g., FID-like) and qualitative artifact inspection, consistent with generative anonymization objectives [1], [6].
6.5 Ablation studies (to establish causal evidence)
	Remove L_"id" : measure identity leakage increase (validates privacy term).
	Remove L_"expr" : measure FER drop / posterior drift (validates utility term).
	Conditioning variants (landmarks vs parsing vs combined): identify which cues best preserve expression while limiting identity reconstruction.
	Detector/alignment variants (MTCNN vs RetinaFace): quantify preprocessing sensitivity [8], [9].
	Adaptive attacker strength sweep: quantify privacy robustness under increasingly capable attackers [5].
Logic bridge: These experiments produce not only performance numbers but evidence for why each component is necessary, enabling a defensible conclusion.
________________________________________
7. Project Plan and Milestones (deliverable-driven)
	M1 — Preprocessing + baselines: implement alignment contract with MTCNN/RetinaFace; reproduce pixel obfuscation and k-Same baselines [3], [8], [9], [14].
	M2 — FER teacher: train/calibrate expression model using FER-2013 and optionally FER+ distributions [10], [11].
	M3 — Identity attacker: set up ArcFace/FaceNet attacker evaluation harness [7], [15].
	M4 — Anonymizer training: pix2pix conditional GAN anonymizer [6] + realism/reconstruction.
	M5 — Full objective: add identity suppression and expression distillation [7], [11], [12].
	M6 — Adaptive attacker + Pareto analysis: adaptive attacker evaluation and privacy–utility frontier report [5].
	M7 — Final report: ablation-backed conclusions, limitations, and recommended operating points.
________________________________________
8. Risks and Mitigations
	GAN instability / mode collapse: start from pix2pix standard training practices [6]; add reconstruction regularization; early stopping based on utility + realism metrics.
	FER label noise: prefer FER+ distribution training to reduce noise sensitivity and use posterior-based consistency instead of hard-label constraints [11].
	Privacy overclaiming: require adaptive attacker evaluation and report both fixed and adaptive results [5].
	Preprocessing confounds: enforce crop contract and explicitly report detector/alignment sensitivity [8], [9].
________________________________________
9. Responsible Research Considerations
This project improves privacy protection but could be misused for evasion. We will:
	keep the evaluation protocol explicit and attacker-aligned;
	report failure cases and non-guarantees;
	restrict conclusions to measured threat models (fixed/adaptive recognizers) rather than absolute anonymity;
	follow dataset licensing and consent requirements.
________________________________________
References (IEEE style)
[1] H. Hukkelås, R. Mester, and F. Lindseth, “DeepPrivacy: A Generative Adversarial Network for Face Anonymization,” arXiv preprint arXiv:1909.04538, 2019. 
[2] M. Maximov, I. Elezi, and L. Leal-Taixé, “CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020. 
[3] E. M. Newton, L. Sweeney, and B. Malin, “Preserving Privacy by De-identifying Facial Images,” IEEE Trans. Knowledge and Data Engineering, 2005. 
[4] R. Gross, L. Sweeney, J. Cohn, F. de la Torre, and S. Baker, “Face De-identification,” in Protecting Privacy in Video Surveillance, 2009 (and related technical reports). 
[5] “Face De-identification: State-of-the-art Methods and Evaluation,” arXiv, 2024. 
[6] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-Image Translation with Conditional Adversarial Networks,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2017. 
[7] J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia, and S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019. 
[8] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks,” arXiv preprint arXiv:1604.02878, 2016. 
[9] J. Deng, J. Guo, Y. Zhou, J. Yu, I. Kotsia, and S. Zafeiriou, “RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020. 
[10] I. J. Goodfellow et al., “Challenges in Representation Learning: A report on three machine learning contests,” 2013 (includes FER-2013 description). 
[11] E. Barsoum, C. Zhang, C. Canton Ferrer, and Z. Zhang, “Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution,” arXiv preprint arXiv:1608.01041, 2016. 
[12] G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv preprint arXiv:1503.02531, 2015. 
[13] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016. 
[14] R. Gross, L. Sweeney, F. de la Torre, and S. Baker, “Integrating Utility into Face De-identification,” in Proc. Int. Conf. Privacy, Security and Trust / related Springer proceedings, 2005 (k-Same-Select). 
[15] F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A Unified Embedding for Face Recognition and Clustering,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015

