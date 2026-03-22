# BASICS-CDSS v2.1.0 - Quick Reference Card

**Print this page and keep it on your desk while updating your manuscript!**

---

## 📋 At a Glance

| Component | Status | Files | Functions | Figures |
|-----------|--------|-------|-----------|---------|
| Clinical Utility | ✅ | 1 | 5 | 5 (1 × 3D) |
| Fairness | ✅ | 1 | 6 | 5 |
| Conformal Prediction | ✅ | 1 | 5 | 4 (1 × 3D) |
| Visualization | ✅ | 1 | 14 | All above |
| Documentation | ✅ | 3 | - | - |
| **TOTAL** | ✅ | **7** | **30** | **21** |

---

## 🚀 Quick Start Commands

### Generate All Figures (for manuscript)
```bash
cd D:\PhD\Manuscript\GitHub\BASICS-CDSS\examples
python generate_clinical_metrics_figures.py --n-samples 500 --output-dir manuscript_figures
```

### Generate Specific Type
```bash
# Clinical utility only
python generate_clinical_metrics_figures.py --utility-only

# Fairness only
python generate_clinical_metrics_figures.py --fairness-only

# Conformal only
python generate_clinical_metrics_figures.py --conformal-only
```

---

## 📝 Manuscript Update Checklist

### ☐ Methods Section (3 subsections)
- [ ] Add "Clinical Utility Assessment" (copy from MANUSCRIPT_UPDATES.md)
- [ ] Add "Fairness and Bias Assessment"
- [ ] Add "Uncertainty Quantification with Conformal Prediction"
- [ ] Add equations and formulas
- [ ] Add references 1-10

### ☐ Results Section (3 subsections)
- [ ] Add "Clinical Utility Results" (fill in [X.X] with your numbers)
- [ ] Add "Fairness Assessment Results"
- [ ] Add "Uncertainty Quantification Results"
- [ ] Create Table X (Fairness Metrics)

### ☐ Figures
- [ ] Figure X: Decision Curve (HIGH PRIORITY)
- [ ] Figure Y: Fairness Radar (HIGH PRIORITY)
- [ ] Figure Z: Conformal Prediction Sets
- [ ] Figure W: Coverage Validation
- [ ] Write captions for all figures

### ☐ Discussion Section (3 subsections)
- [ ] Add "Clinical Utility and Deployment Readiness"
- [ ] Add "Fairness, Equity, and Ethical AI"
- [ ] Add "Uncertainty Quantification and Clinical Trust"

### ☐ Limitations
- [ ] Add "Limitations of Phase 1 Clinical Metrics"

### ☐ Conclusion
- [ ] Update with Phase 1 results summary

### ☐ References
- [ ] Add 10 new references (list in MANUSCRIPT_UPDATES.md)

---

## 📊 Where to Get Your Numbers

After running generation script, look for these in the console output:

### Clinical Utility
```
Look for: "Useful threshold range: (X.XX, X.XX)"
Look for: "Model NNT: X.X (ARR: XX.X%)"
```
Check `clinical_impact_0.3.pdf` for PPV, NPV values

### Fairness
```
Look for: "Overall Fair: True/False"
Look for: "Failed Criteria: [...]"
```
Measure differences from generated fairness PDFs

### Conformal Prediction
```
Look for: "Target Coverage: 90.0%"
Look for: "Average Set Size: X.XX"
Look for: "Singleton Sets: X/Y"
```

---

## 🎯 Minimum Figures to Add

**For FDA/Publication, add at least these 2:**

1. **Decision Curve** (`decision_curve.pdf`)
   - Shows clinical utility vs thresholds
   - Compare to treat-all and treat-none

2. **Fairness Radar** (`fairness_radar_race.pdf`)
   - Shows all 5 fairness metrics
   - Visual summary of equity

**Recommended: Add these 2 more:**

3. **Clinical Impact** (`clinical_impact_0.3.pdf`)
   - Classification breakdown
   - PPV, NPV, NNS

4. **Prediction Sets** (`prediction_set_sizes.pdf`)
   - Uncertainty quantification
   - Coverage guarantee validation

---

## 📚 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **MANUSCRIPT_UPDATES.md** | Ready-to-copy text for paper | NOW - updating manuscript |
| **CLINICAL_METRICS_GUIDE.md** | Detailed technical guide | Reference for understanding metrics |
| **IMPLEMENTATION_SUMMARY.md** | Complete project summary | Overview of everything done |
| **QUICK_REFERENCE.md** | This file! | Quick lookup while working |

---

## 🔑 Key Formulas (for Methods section)

### Net Benefit
```
NB(pt) = (TP/N) - (FP/N) × [pt/(1-pt)]
```

### Number Needed to Treat
```
NNT = 1 / ARR
ARR = |CER - EER|
```

### Coverage Guarantee (Conformal)
```
P(Y ∈ C(X)) ≥ 1 - α
```

### Fairness Metrics
```
Demographic Parity: P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
Equalized Odds: P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b)
Disparate Impact: 0.8 ≤ DI ≤ 1.25
```

---

## 📖 Key References (Copy to Bibliography)

**Clinical Utility:**
1. Vickers AJ, Elkin EB. Decision curve analysis... Medical Decision Making. 2006;26(6):565-574.
2. Vickers AJ, et al. Net benefit approaches... BMJ. 2016;352:i6.

**Fairness:**
3. Hardt M, et al. Equality of opportunity... NIPS. 2016.
4. Obermeyer Z, et al. Dissecting racial bias... Science. 2019;366(6464):447-453.

**Conformal Prediction:**
5. Vovk V, et al. Algorithmic Learning in a Random World. Springer. 2005.
6. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction. arXiv:2107.07511. 2021.

(See MANUSCRIPT_UPDATES.md for complete list of 10 references)

---

## ⏱️ Time Estimates

| Task | Time | Priority |
|------|------|----------|
| Copy Methods sections | 45 min | HIGH |
| Fill Results with numbers | 45 min | HIGH |
| Write Discussion additions | 45 min | HIGH |
| Add figures + captions | 30 min | HIGH |
| Create Table X | 30 min | MEDIUM |
| Add all references | 15 min | MEDIUM |
| Review and polish | 30 min | LOW |
| **TOTAL (minimum)** | **~4 hours** | - |

---

## ✅ Quality Check Before Submission

Before submitting manuscript, verify:

- [ ] All [X.X] placeholders replaced with actual numbers
- [ ] Figure numbers (X, Y, Z) match your manuscript
- [ ] Table X created with your actual fairness metrics
- [ ] All 10 new references added to bibliography
- [ ] Figures are 300 DPI and readable
- [ ] Fairness discussion matches your results (PASS/FAIL)
- [ ] Equations formatted correctly
- [ ] Co-authors reviewed fairness interpretation

---

## 🆘 If You Get Stuck

**Problem:** Can't find where to insert new sections
**Solution:** Look for your current "Evaluation" or "Metrics" section, add subsections there

**Problem:** Don't know what numbers to fill in [X.X]
**Solution:** Run the generation script and look at console output (see "Where to Get Numbers" above)

**Problem:** Fairness metrics failed, unsure how to write it
**Solution:** See MANUSCRIPT_UPDATES.md section "IF FAILED SOME FAIRNESS CRITERIA" for template

**Problem:** Confused about a metric's meaning
**Solution:** Check CLINICAL_METRICS_GUIDE.md for detailed explanations with clinical examples

**Problem:** Need to understand the code
**Solution:** All functions have detailed docstrings - use `help(function_name)` in Python

---

## 🎯 Success Criteria

**Your manuscript update is complete when:**

✅ Methods section has 3 new subsections (Clinical Utility, Fairness, Conformal)
✅ Results section has 3 new subsections with YOUR numbers (not [X.X])
✅ At least 2 new figures added (Decision Curve + Fairness Radar minimum)
✅ Discussion mentions all 3 components (utility, fairness, uncertainty)
✅ Limitations mentions temporal validation and external validation
✅ 10 new references added to bibliography
✅ Table showing fairness metrics across groups

---

## 📞 Quick Help

**File locations:**
- Ready-to-use text: `docs/MANUSCRIPT_UPDATES.md`
- Generated figures: `clinical_test/` or `manuscript_figures/`
- Detailed guide: `docs/CLINICAL_METRICS_GUIDE.md`

**Python help:**
```python
from basics_cdss.clinical_metrics import calculate_net_benefit
help(calculate_net_benefit)  # Shows detailed documentation
```

---

**🎉 You've got this! Phase 1 is complete and ready to enhance your manuscript.**

**Estimated time to update manuscript: 4-5 hours**

**Expected outcome: Publication-ready paper with comprehensive Phase 1 Clinical Metrics**

---

Print Date: 2025-01-25 | Version: 2.1.0 | BASICS-CDSS Phase 1 Complete ✅
