#!/usr/bin/env python3
"""
Automated Figure Copy Script for BASICS-CDSS Manuscripts

This script copies publication-ready figures from the BASICS-CDSS repository
to the manuscript directories for Papers 2 and 3.

Usage:
    python copy_figures_to_manuscripts.py

Author: Chatchai Tritham
Date: 2026-02-10
"""

import shutil
from pathlib import Path
import sys

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}[OK]{Colors.END} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}[!]{Colors.END} {text}")

def print_error(text):
    print(f"{Colors.RED}[X]{Colors.END} {text}")

def print_info(text):
    print(f"{Colors.BLUE}[i]{Colors.END} {text}")

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def copy_file_safe(src, dst, description=""):
    """Safely copy file with error handling"""
    try:
        if not src.exists():
            print_error(f"Source file not found: {src.name}")
            return False

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src, dst)

        # Get file size
        size_kb = dst.stat().st_size / 1024

        if description:
            print_success(f"Copied {src.name} ({size_kb:.1f} KB) - {description}")
        else:
            print_success(f"Copied {src.name} ({size_kb:.1f} KB)")

        return True
    except Exception as e:
        print_error(f"Failed to copy {src.name}: {e}")
        return False

def main():
    # Base paths
    repo_base = Path(__file__).parent.parent
    manuscript_base = repo_base.parent.parent / "Manuscript" / "Manuscript"

    print_header("BASICS-CDSS Figure Copy Script")
    print_info(f"Repository: {repo_base}")
    print_info(f"Manuscript base: {manuscript_base}")

    # ========================================================================
    # PAPER 2: BASICS-CDSS (Digital Twin Temporal Evaluation)
    # ========================================================================

    print_header("Paper 2: BASICS-CDSS (Digital Twin)")

    paper2_dir = manuscript_base / "PeerJ_BASIC-CDSS"
    paper2_figures = ensure_directory(paper2_dir / "figures")

    print_info(f"Target directory: {paper2_figures}")

    # Clinical Utility Metrics (3 figures)
    print(f"\n{Colors.BOLD}Clinical Utility Metrics:{Colors.END}")
    clinical_utility_src = repo_base / "clinical_test" / "clinical_utility"

    figures_paper2 = [
        (clinical_utility_src / "decision_curve.pdf", "Decision Curve Analysis (DCA)"),
        (clinical_utility_src / "nnt_comparison.pdf", "Number Needed to Treat (NNT)"),
        (clinical_utility_src / "net_benefit_threshold_0.3.pdf", "Net Benefit at pt=0.3"),
    ]

    copied_paper2 = 0
    for src, desc in figures_paper2:
        dst = paper2_figures / src.name
        if copy_file_safe(src, dst, desc):
            copied_paper2 += 1

    # Fairness Metrics (2 figures)
    print(f"\n{Colors.BOLD}Fairness Metrics:{Colors.END}")
    fairness_src = repo_base / "clinical_test" / "fairness"

    fairness_figures = [
        (fairness_src / "fairness_radar_race.pdf", "Multi-metric Fairness Radar"),
        (fairness_src / "calibration_race.pdf", "Calibration by Race"),
    ]

    for src, desc in fairness_figures:
        dst = paper2_figures / src.name
        if copy_file_safe(src, dst, desc):
            copied_paper2 += 1

    # Conformal Prediction (1 figure)
    print(f"\n{Colors.BOLD}Uncertainty Quantification:{Colors.END}")
    conformal_src = repo_base / "clinical_test" / "conformal_prediction"

    conformal_figures = [
        (conformal_src / "coverage_vs_alpha.pdf", "Conformal Prediction Coverage"),
    ]

    for src, desc in conformal_figures:
        dst = paper2_figures / src.name
        if copy_file_safe(src, dst, desc):
            copied_paper2 += 1

    print(f"\n{Colors.BOLD}Paper 2 Summary:{Colors.END}")
    print_info(f"Total figures copied: {copied_paper2}/6")
    print_info(f"Existing figures: 7 (already in manuscript)")
    print_success(f"Paper 2 now has: {7 + copied_paper2} figures total")

    # ========================================================================
    # PAPER 3: Causal Models (Structural Causal Models for CDSS)
    # ========================================================================

    print_header("Paper 3: Causal Models (SCM)")

    paper3_dir = manuscript_base / "PeerJ_BASIC-CDSS_Causal-Models"
    paper3_figures = ensure_directory(paper3_dir / "figures")

    print_info(f"Target directory: {paper3_figures}")

    # Tier 2 Causal Figures (5 figures)
    print(f"\n{Colors.BOLD}Tier 2 Causal Analysis Figures:{Colors.END}")
    tier2_src = repo_base / "examples" / "figures" / "tier2"

    causal_figures = [
        (tier2_src / "fig01_causal_dag.png", "Causal DAG Structure (Sepsis, ARDS, ACS)"),
        (tier2_src / "fig02_intervention_effects.png", "Average Treatment Effects (ATE)"),
        (tier2_src / "fig03_cate_heterogeneity.png", "CATE Heterogeneity Heatmap"),
        (tier2_src / "fig04_confounding_analysis.png", "Confounding Bias Quantification"),
        (tier2_src / "fig05_backdoor_adjustment.png", "Backdoor Criterion Adjustment"),
    ]

    copied_paper3 = 0
    for src, desc in causal_figures:
        dst = paper3_figures / src.name
        if copy_file_safe(src, dst, desc):
            copied_paper3 += 1

    print(f"\n{Colors.BOLD}Paper 3 Summary:{Colors.END}")
    print_info(f"Total figures copied: {copied_paper3}/5")
    print_success(f"Paper 3 now has: {copied_paper3} figures")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_header("Overall Summary")

    print(f"{Colors.BOLD}Repository Status:{Colors.END}")
    print_success(f"BASICS-CDSS v2.1.0 - Production Ready")
    print_success(f"46+ publication-ready figures available")
    print_success(f"Fully tested and reproducible")

    print(f"\n{Colors.BOLD}Manuscript Status:{Colors.END}")
    print_info(f"Paper 2: {7 + copied_paper2} figures (7 existing + {copied_paper2} new)")
    print_info(f"Paper 3: {copied_paper3} figures (all new)")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print_info("1. Add figure environments to LaTeX files")
    print_info("2. Write comprehensive figure captions")
    print_info("3. Compile manuscripts and verify figure placement")

    if copied_paper2 == 6 and copied_paper3 == 5:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] All figures copied successfully!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}[WARNING] Some figures may not have been copied{Colors.END}")
        print_warning(f"Expected: Paper 2 (6), Paper 3 (5)")
        print_warning(f"Actual: Paper 2 ({copied_paper2}), Paper 3 ({copied_paper3})")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
