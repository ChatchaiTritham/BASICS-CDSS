# Quick Start: Manuscript Integration Guide

**Date**: 2026-02-10
**Version**: 2.1.0
**Status**: All figures copied, captions ready

---

## Summary

✅ **All 11 figures successfully copied to manuscript directories**
- Paper 2: 6 clinical metrics figures added (now 13 total figures)
- Paper 3: 5 causal analysis figures added (now 5 total figures)

---

## Step-by-Step Integration

### Paper 2: BASICS-CDSS (Digital Twin)

**Location**: `D:\PhD\Manuscript\Manuscript\PeerJ_BASIC-CDSS\`

#### 1. Verify Figures (Already Done ✅)

```bash
ls figures/
# Should show:
# - decision_curve.pdf (49.5 KB)
# - nnt_comparison.pdf (43.0 KB)
# - net_benefit_threshold_0.3.pdf (40.7 KB)
# - fairness_radar_race.pdf (48.2 KB)
# - calibration_race.pdf (53.1 KB)
# - coverage_vs_alpha.pdf (44.4 KB)
```

#### 2. Add Figures to LaTeX

**Option A**: Copy from captions file:

```bash
# LaTeX file already created with all captions
cat FIGURE_CAPTIONS_CLINICAL_METRICS.tex
```

**Option B**: Add manually to `latex/sections/04_results.tex`:

```latex
% After existing Figure 7 (temporal_consistency.pdf)

% Clinical Utility Section
\subsection{Clinical Utility Metrics}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/decision_curve.pdf}
\caption{Decision Curve Analysis for Clinical Utility Assessment. [See FIGURE_CAPTIONS_CLINICAL_METRICS.tex for full caption]}
\label{fig:decision_curve}
\end{figure}

% ... repeat for figures 9-13
```

#### 3. Compile LaTeX

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

### Paper 3: Causal Models (SCM)

**Location**: `D:\PhD\Manuscript\Manuscript\PeerJ_BASIC-CDSS_Causal-Models\`

#### 1. Verify Figures (Already Done ✅)

```bash
ls figures/
# Should show:
# - fig01_causal_dag.png (204.2 KB)
# - fig02_intervention_effects.png (138.3 KB)
# - fig03_cate_heterogeneity.png (115.6 KB)
# - fig04_confounding_analysis.png (189.0 KB)
# - fig05_backdoor_adjustment.png (189.0 KB)
```

#### 2. Convert to PeerJ Template (Required)

**Current**: Nature Machine Intelligence style (`twocolumn`)
**Need**: PeerJ CS style (`wlpeerj`)

```bash
# Create conversion script
cd D:/PhD/Manuscript/Manuscript/PeerJ_BASIC-CDSS_Causal-Models
python ../PeerJ_BASIC-CDSS/convert_to_peerj.py  # Adapt from Paper 2
```

**Manual changes needed in `latex/main.tex`**:

```latex
% OLD:
\documentclass[twocolumn]{article}

% NEW:
\documentclass[fleqn,10pt]{wlpeerj}
```

#### 3. Add Figure Environments

Add to `latex/sections/03_methods.tex` and `04_experiments.tex`:

```latex
% In Methods section
\subsection{Structural Causal Models}

\begin{figure*}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig01_causal_dag.png}
\caption{Causal Directed Acyclic Graphs (DAGs) for Three Clinical Domains. [See FIGURE_CAPTIONS_CAUSAL.tex for full caption]}
\label{fig:causal_dag}
\end{figure*}

% In Results section
\subsection{Average Treatment Effects}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig02_intervention_effects.png}
\caption{Average Treatment Effects (ATE) Forest Plot. [See FIGURE_CAPTIONS_CAUSAL.tex for full caption]}
\label{fig:intervention_effects}
\end{figure}

% ... repeat for figures 3-5
```

#### 4. Compile LaTeX

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Figure Caption Integration

### For Paper 2

**File created**: `FIGURE_CAPTIONS_CLINICAL_METRICS.tex`

**Usage**:

```latex
% Option 1: Include entire file
\input{FIGURE_CAPTIONS_CLINICAL_METRICS.tex}

% Option 2: Copy individual captions
% Open file and copy the specific \begin{figure}...\end{figure} blocks
```

**Figures included** (6 new):
- Figure 8: Decision Curve Analysis (DCA)
- Figure 9: Number Needed to Treat (NNT)
- Figure 10: Net Benefit at threshold 0.30
- Figure 11: Fairness Radar Chart (Race)
- Figure 12: Calibration by Race
- Figure 13: Conformal Prediction Coverage

---

### For Paper 3

**File created**: `FIGURE_CAPTIONS_CAUSAL.tex`

**Usage**:

```latex
% Option 1: Include entire file
\input{FIGURE_CAPTIONS_CAUSAL.tex}

% Option 2: Copy individual captions
% Open file and copy the specific \begin{figure}...\end{figure} blocks
```

**Figures included** (5 total):
- Figure 1: Causal DAG Structure (Sepsis, ARDS, ACS)
- Figure 2: Average Treatment Effects (ATE) Forest Plot
- Figure 3: CATE Heterogeneity Heatmap
- Figure 4: Confounding Bias Quantification
- Figure 5: Backdoor Criterion Adjustment

---

## Manuscript Status

### Paper 1: SynDX-Hybrid ✅ 100% Complete
- **Status**: Ready for submission
- **Figures**: 7 main + 5 supplementary = 12 total
- **Word count**: ~9,300 words
- **Action**: Submit to PeerJ CS immediately

### Paper 2: BASICS-CDSS ⚠️ 90% Complete
- **Status**: 4-5 hours to ready
- **Figures**: 13 total (7 existing + 6 new clinical metrics)
- **Word count**: ~12,250 words
- **Remaining tasks**:
  1. ✅ Copy figures (DONE)
  2. ✅ Write captions (DONE)
  3. ⏳ Add to LaTeX (2 hours)
  4. ⏳ Create references.bib (2 hours)
  5. ⏳ Compile and verify (30 min)
- **Action**: Complete by Feb 15-17

### Paper 3: Causal Models ⚠️ 85% Complete
- **Status**: 3-4 hours to ready
- **Figures**: 5 total (all new)
- **Word count**: ~17,150 words
- **Remaining tasks**:
  1. ✅ Copy figures (DONE)
  2. ✅ Write captions (DONE)
  3. ⏳ Convert template to PeerJ (30 min)
  4. ⏳ Add to LaTeX (1 hour)
  5. ⏳ Create references.bib (1 hour)
  6. ⏳ Compile and verify (30 min)
- **Action**: Complete by Feb 17-19

---

## Automated Script Usage

### Copy All Figures (Already Run ✅)

```bash
cd D:/PhD/Manuscript/GitHub/BASICS-CDSS
python scripts/copy_figures_to_manuscripts.py
```

**Output**:
```
[SUCCESS] All figures copied successfully!
Paper 2: 13 figures (7 existing + 6 new)
Paper 3: 5 figures (all new)
```

### Regenerate Figures (Optional)

If you need to regenerate figures from scratch:

```bash
cd D:/PhD/Manuscript/GitHub/BASICS-CDSS/examples

# Clinical metrics
python generate_clinical_metrics_figures.py --n-samples 500

# Performance metrics
python generate_performance_figures.py

# XAI figures
python generate_xai_figures.py
```

---

## Repository Improvements Summary

### ✅ Completed

1. **README.md** updated:
   - Fixed repository URL (BASIC-CDSS → BASICS-CDSS)
   - Added installation instructions
   - Added reproducibility section
   - Added BibTeX citation

2. **CITATION.cff** updated:
   - Version 2.1.0
   - Date 2026-02-10
   - Correct repository URL

3. **pyproject.toml** enhanced:
   - Python ≥3.9 support
   - Author emails added
   - Keywords and classifiers
   - Project URLs (homepage, docs, issues)
   - Optional dependencies (dev, notebooks)

4. **Script created**: `scripts/copy_figures_to_manuscripts.py`
   - Automated figure copying
   - Colored terminal output
   - Error handling
   - Summary statistics

5. **Figure captions created**:
   - `FIGURE_CAPTIONS_CLINICAL_METRICS.tex` (6 figures, ~4,500 words)
   - `FIGURE_CAPTIONS_CAUSAL.tex` (5 figures, ~4,200 words)

---

## Next Steps

### Today (Feb 10)

1. ✅ Repository improvements (DONE)
2. ✅ Copy figures (DONE)
3. ✅ Write captions (DONE)
4. ⏳ **Push updates to GitHub**:

```bash
cd D:/PhD/Manuscript/GitHub/BASICS-CDSS
git add README.md CITATION.cff pyproject.toml scripts/
git commit -m "Update repository metadata and add manuscript integration tools

- Fix repository URL in README (BASICS-CDSS)
- Update CITATION.cff to v2.1.0
- Enhance pyproject.toml with metadata and URLs
- Add automated figure copy script
- Support Python 3.9+
"
git push
```

### Tomorrow (Feb 11)

**Paper 2**:
1. Add figure environments to `latex/sections/04_results.tex` (1 hour)
2. Create `latex/references.bib` (2 hours)
3. Compile LaTeX (30 min)
4. **Submit to PeerJ CS** ✅

### Feb 12-17

**Paper 3**:
1. Convert template to PeerJ (30 min)
2. Add figure environments (1 hour)
3. Create `latex/references.bib` (1 hour)
4. Compile LaTeX (30 min)
5. **Submit to PeerJ CS** ✅

---

## Contact

**Repository**: https://github.com/ChatchaiTritham/BASICS-CDSS

**Issues**: https://github.com/ChatchaiTritham/BASICS-CDSS/issues

**Author**: Chatchai Tritham (chatchait66@nu.ac.th)

---

**Status**: ✅ Repository 100% ready, Manuscripts 85-90% ready, All improvements complete
