"""
Quick smoke test for the full analysis pipeline.
Run from backend/ with: python test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from core.structural_analyzer import StructuralAnalyzer
from core.statistical_engine import StatisticalEngine
from core.model_recommender import ModelRecommender
from core.insight_generator import InsightGenerator


def make_test_dataframe():
    """Creates a realistic test dataset with common issues."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.exponential(50000, n),          # skewed
        'score': np.random.normal(0.5, 0.1, n),
        'category': np.random.choice(['A', 'B', 'C'], n, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'churn': np.random.choice([0, 1], n, p=[0.9, 0.1]),  # imbalanced
        'duplicate_flag': np.ones(n),                         # constant
    })

    # Inject missing values
    df.loc[np.random.choice(n, 60, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), 'category'] = np.nan

    # Add duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    return df


def run():
    print("=" * 60)
    print("DataSense Pipeline Smoke Test")
    print("=" * 60)

    df = make_test_dataframe()
    print(f"\n✓ Test dataset created: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 1: Structural Analysis
    print("\n[1/4] Running structural analysis...")
    analyzer = StructuralAnalyzer()
    blueprint = analyzer.analyze(df)
    basic = blueprint['basic_info']
    print(f"  rows={basic['rows']}, cols={basic['columns']}, "
          f"missing={basic['missing_percentage']}%, "
          f"duplicates={basic['duplicate_rows']}")
    print(f"  structure type: {blueprint['data_structure']['type']}")
    print(f"  quality issues found: {len(blueprint['quality_issues'])}")
    print(f"  target candidates: {[t['column'] for t in blueprint['target_candidates']]}")
    print("  ✓ Structural analysis complete")

    # Step 2: Statistical Engine
    print("\n[2/4] Running statistical engine...")
    engine = StatisticalEngine()
    stats = engine.analyze(df, blueprint)
    print(f"  red flags: {len(stats['red_flags'])}")
    print(f"  warnings: {len(stats['warnings'])}")
    print(f"  correlation pairs computed: {len(stats['correlations'])}")
    print(f"  outlier columns: {len(stats['outlier_summary'])}")
    print("  ✓ Statistical analysis complete")

    # Step 3: Model Recommender
    print("\n[3/4] Running model recommender...")
    recommender = ModelRecommender()
    recommendations = recommender.recommend(blueprint, stats)
    print(f"  task type: {recommendations['task_type']}")
    print(f"  primary model: {recommendations['primary_model']}")
    print(f"  confidence: {recommendations['confidence']}")
    print(f"  preprocessing steps: {len(recommendations['preprocessing_steps'])}")
    print("  ✓ Model recommendations complete")

    # Step 4: Insight Generator (no LLM for smoke test)
    print("\n[4/4] Generating insights (LLM disabled for speed)...")
    generator = InsightGenerator(use_llm=False)
    insights = generator.generate_insights(blueprint, stats, recommendations)
    print(f"  executive summary: {insights['executive_summary'][:100]}...")
    print(f"  critical: {insights['severity_breakdown']['critical']}")
    print(f"  high: {insights['severity_breakdown']['high']}")
    print(f"  medium: {insights['severity_breakdown']['medium']}")
    print(f"  quick wins: {len(insights['quick_wins'])}")
    print("  ✓ Insights complete")

    print("\n" + "=" * 60)
    print("✅ ALL PIPELINE STAGES PASSED")
    print("=" * 60)

    print("\nSample quick wins:")
    for i, win in enumerate(insights['quick_wins'], 1):
        print(f"  {i}. {win}")

    print("\nSample critical insights:")
    for ins in insights['critical_insights']:
        print(f"  [{ins['severity'].upper()}] {ins['headline']}")
        print(f"    → {ins['what_to_do']}")


if __name__ == "__main__":
    run()
