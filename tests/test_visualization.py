"""
Test visualization module imports and function availability
Tests all visualization functions without generating actual plots
"""
import sys
import inspect

def test_visualization_imports():
    """Test importing visualization modules."""
    print("=" * 80)
    print("Visualization Module Import Test")
    print("=" * 80)
    print()

    # Test performance plots
    print("1. Performance Plots Module")
    print("-" * 80)
    try:
        from basics_cdss.visualization import performance_plots
        funcs = [name for name, obj in inspect.getmembers(performance_plots)
                 if inspect.isfunction(obj) and not name.startswith('_')]
        print(f"[OK] performance_plots imported - {len(funcs)} functions")
        for func in funcs:
            print(f"     - {func}")
        perf_count = len(funcs)
    except Exception as e:
        print(f"[FAIL] performance_plots import failed: {e}")
        perf_count = 0

    print()

    # Test advanced charts
    print("2. Advanced Charts Module")
    print("-" * 80)
    try:
        from basics_cdss.visualization import advanced_charts
        funcs = [name for name, obj in inspect.getmembers(advanced_charts)
                 if inspect.isfunction(obj) and not name.startswith('_')]
        print(f"[OK] advanced_charts imported - {len(funcs)} functions")
        for func in funcs:
            print(f"     - {func}")
        adv_count = len(funcs)
    except Exception as e:
        print(f"[FAIL] advanced_charts import failed: {e}")
        adv_count = 0

    print()

    # Test direct imports from visualization package
    print("3. Direct Package Imports")
    print("-" * 80)

    performance_plot_funcs = [
        'plot_confusion_matrix',
        'plot_roc_curve',
        'plot_pr_curve',
        'plot_sensitivity_specificity_curve',
        'plot_threshold_analysis',
        'plot_multi_model_roc',
        'plot_metrics_comparison_bar',
        'plot_multi_class_confusion_matrix',
    ]

    advanced_chart_funcs = [
        'plot_3d_performance_surface',
        'plot_contour_performance',
        'plot_stratified_heatmap',
        'plot_radar_chart',
        'plot_multi_radar_comparison',
        'plot_parallel_coordinates',
        'plot_3d_scatter_performance',
    ]

    all_funcs = performance_plot_funcs + advanced_chart_funcs

    successful_imports = 0
    failed_imports = 0

    for func_name in all_funcs:
        try:
            exec(f"from basics_cdss.visualization import {func_name}")
            print(f"[OK] {func_name}")
            successful_imports += 1
        except Exception as e:
            print(f"[FAIL] {func_name}: {e}")
            failed_imports += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Performance plots functions: {perf_count}")
    print(f"Advanced charts functions:   {adv_count}")
    print(f"Direct imports successful:   {successful_imports}/{len(all_funcs)}")
    print(f"Direct imports failed:       {failed_imports}/{len(all_funcs)}")
    print(f"Total visualization functions: {successful_imports}")
    print()

    if failed_imports == 0:
        print("[SUCCESS] All visualization functions are importable!")
    assert failed_imports == 0, "Some visualization functions failed to import"

if __name__ == '__main__':
    try:
        test_visualization_imports()
    except AssertionError:
        sys.exit(1)
