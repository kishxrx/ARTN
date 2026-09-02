#!/usr/bin/env python3
"""
Compare LR Experiment Results - EVIDENCE-BASED ANALYSIS

Analyzes results from multiple lr_experiment.py runs and ranks hypotheses.
Uses MULTI-SIGNAL comparison, not just training loss.

Usage:
    python experiments/compare_lr_results.py --results_dir results/lr_experiments
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# ============================================================
# Signal Extraction Functions
# ============================================================

def load_experiment_results(exp_dir):
    """Load results from experiment directory"""
    results_file = Path(exp_dir) / "results.json"
    
    if not results_file.exists():
        print(f"ERROR: No results.json found in {exp_dir}")
        return None
    
    with open(results_file) as f:
        return json.load(f)

def extract_trajectory(metrics_history: List[dict]) -> dict:
    """Extract loss trajectory statistics"""
    
    losses = [m["train_loss"] for m in metrics_history]
    
    if len(losses) < 2:
        return {
            "initial": losses[0] if losses else 0,
            "final": losses[-1] if losses else 0,
            "best": min(losses) if losses else 0,
            "trend": "insufficient_data"
        }
    
    # Calculate trend (linear slope)
    steps = np.arange(len(losses))
    slope, intercept = np.polyfit(steps, losses, 1)
    
    # Calculate reduction
    reduction = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
    
    # Check if still descending
    recent_losses = losses[-3:] if len(losses) >= 3 else losses[-1:]
    still_descending = recent_losses[-1] < recent_losses[0]
    
    return {
        "initial": float(losses[0]),
        "final": float(losses[-1]),
        "best": float(min(losses)),
        "reduction_pct": float(reduction),
        "slope": float(slope),
        "still_descending": still_descending,
        "trajectory": losses
    }

def extract_perplexity(metrics_history: List[dict]) -> dict:
    """Extract perplexity statistics"""
    
    ppls = [m.get("perplexity", 0) for m in metrics_history]
    
    if not ppls:
        return {"initial": 0, "final": 0, "best": 0, "improvement": 0}
    
    reduction = (ppls[0] - ppls[-1]) / ppls[0] * 100 if ppls[0] > 0 else 0
    
    return {
        "initial": float(ppls[0]),
        "final": float(ppls[-1]),
        "best": float(min(ppls)),
        "reduction_pct": float(reduction)
    }

def extract_gradient_stats(metrics_history: List[dict]) -> dict:
    """Extract gradient norm statistics"""
    
    grad_norms = [m.get("grad_norm", 0) for m in metrics_history]
    
    if not grad_norms:
        return {"avg": 0, "min": 0, "max": 0, "vanishing": False, "exploding": False}
    
    avg_norm = float(np.mean(grad_norms))
    
    return {
        "avg": avg_norm,
        "min": float(min(grad_norms)),
        "max": float(max(grad_norms)),
        "vanishing": avg_norm < 1e-6,
        "exploding": avg_norm > 100,
        "healthy": 1e-6 <= avg_norm <= 100,
        "trajectory": grad_norms
    }

def extract_lr_behavior(metrics_history: List[dict]) -> dict:
    """Extract learning rate behavior"""
    
    lrs = [m.get("lr", 0) for m in metrics_history]
    
    if not lrs:
        return {"initial": 0, "final": 0, "avg": 0, "decayed": False}
    
    return {
        "initial": float(lrs[0]),
        "final": float(lrs[-1]),
        "avg": float(np.mean(lrs)),
        "decayed": lrs[-1] < lrs[0]
    }

def analyze_generations(samples: List[dict]) -> dict:
    """Analyze generated text quality"""
    
    if not samples:
        return {
            "avg_repetition": 0,
            "max_repetition": 0,
            "coherent": False,
            "samples": []
        }
    
    rep_scores = [s["repetition_score"] for s in samples]
    texts = [s["generated"] for s in samples]
    
    # Check for basic structure markers
    def has_structure(text: str) -> bool:
        text_lower = text.lower()
        
        # Check for common story patterns
        has_capital_start = text[0].isupper() if text else False
        has_period = '.' in text
        has_spaces = ' ' in text
        
        # Check for word diversity
        words = text.split()
        if len(words) < 3:
            return False
        
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words if total_words > 0 else 0
        
        # Structure score: multiple indicators
        structure_count = sum([has_capital_start, has_period, has_spaces])
        
        return structure_count >= 2 and diversity > 0.3
    
    coherent_count = sum(1 for text in texts if has_structure(text))
    
    # Analyze word patterns
    word_patterns = []
    for text in texts:
        words = text.split()
        if len(words) >= 2:
            # Check for repetition of same word
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeat = max(word_counts.values()) if word_counts else 1
            word_patterns.append(max_repeat / len(words) if words else 0)
    
    return {
        "avg_repetition": float(np.mean(rep_scores)) if rep_scores else 0,
        "max_repetition": float(max(rep_scores)) if rep_scores else 0,
        "min_repetition": float(min(rep_scores)) if rep_scores else 0,
        "coherent_count": coherent_count,
        "coherent_ratio": coherent_count / len(texts) if texts else 0,
        "avg_word_repetition": float(np.mean(word_patterns)) if word_patterns else 0,
        "samples": [
            {
                "prompt": s["prompt"],
                "generated": s["generated"][:100] + "..." if len(s["generated"]) > 100 else s["generated"],
                "repetition": s["repetition_score"]
            }
            for s in samples[:3]  # Store first 3 samples
        ]
    }

# ============================================================
# Comparison Functions
# ============================================================

def compare_trajectories(exp1: dict, exp2: dict) -> str:
    """Compare two experiments' trajectories"""
    
    t1 = exp1["trajectory"]
    t2 = exp2["trajectory"]
    
    # Compare final vs best
    if t2["best"] < t1["best"] * 0.9:  # 10% better
        return "BETTER"
    elif t2["best"] > t1["best"] * 1.1:  # 10% worse
        return "WORSE"
    else:
        return "SIMILAR"

def compare_perplexity(exp1: dict, exp2: dict) -> str:
    """Compare perplexity reduction"""
    
    p1 = exp1["perplexity"]
    p2 = exp2["perplexity"]
    
    if p2["reduction_pct"] > p1["reduction_pct"] + 10:  # 10% more reduction
        return "BETTER"
    elif p2["reduction_pct"] < p1["reduction_pct"] - 10:
        return "WORSE"
    else:
        return "SIMILAR"

def compare_gradients(exp1: dict, exp2: dict) -> str:
    """Compare gradient health"""
    
    g1 = exp1["gradients"]
    g2 = exp2["gradients"]
    
    # Both healthy
    if g1["healthy"] and g2["healthy"]:
        # Compare magnitudes (prefer larger but not exploding)
        if g2["avg"] > g1["avg"] and g2["avg"] < 10:
            return "BETTER"
        elif g2["avg"] < g1["avg"]:
            return "WORSE"
        else:
            return "SIMILAR"
    
    # Health issues
    if g2["vanishing"] or g2["exploding"]:
        return "WORSE"
    elif g1["vanishing"] or g1["exploding"]:
        return "BETTER"
    else:
        return "SIMILAR"

def compare_generations(exp1: dict, exp2: dict) -> str:
    """Compare generation quality"""
    
    g1 = exp1["generations"]
    g2 = exp2["generations"]
    
    signals = []
    
    # Lower repetition is better
    if g2["avg_repetition"] < g1["avg_repetition"] - 5:
        signals.append("repetition_lower")
    elif g2["avg_repetition"] > g1["avg_repetition"] + 5:
        signals.append("repetition_higher")
    
    # Higher coherent ratio is better
    if g2["coherent_ratio"] > g1["coherent_ratio"] + 0.2:
        signals.append("more_coherent")
    elif g2["coherent_ratio"] < g1["coherent_ratio"] - 0.2:
        signals.append("less_coherent")
    
    # Word repetition
    if g2["avg_word_repetition"] < g1["avg_word_repetition"]:
        signals.append("less_word_repetition")
    elif g2["avg_word_repetition"] > g1["avg_word_repetition"]:
        signals.append("more_word_repetition")
    
    # Count positive vs negative
    positive = sum(1 for s in signals if s in ["repetition_lower", "more_coherent", "less_word_repetition"])
    negative = sum(1 for s in signals if s in ["repetition_higher", "less_coherent", "more_word_repetition"])
    
    if positive >= 2 and negative == 0:
        return "BETTER"
    elif negative >= 2 and positive == 0:
        return "WORSE"
    else:
        return "SIMILAR"

# ============================================================
# Hypothesis Evaluation
# ============================================================

def evaluate_h1_lr_too_low(results: Dict[str, dict]) -> Tuple[str, str]:
    """H1: LR too low (0.001 insufficient)"""
    
    if "lr_001_constant" not in results or "lr_003_constant" not in results:
        return "UNKNOWN", "Missing required experiments"
    
    baseline = results["lr_001_constant"]
    higher = results["lr_003_constant"]
    
    signals = []
    
    # Compare trajectory
    traj_result = compare_trajectories(baseline, higher)
    if traj_result == "BETTER":
        signals.append("trajectory_improved")
    
    # Compare perplexity
    ppl_result = compare_perplexity(baseline, higher)
    if ppl_result == "BETTER":
        signals.append("perplexity_improved")
    
    # Compare gradients
    grad_result = compare_gradients(baseline, higher)
    if grad_result == "BETTER":
        signals.append("gradients_healthier")
    
    # Compare generations (MOST IMPORTANT)
    gen_result = compare_generations(baseline, higher)
    if gen_result == "BETTER":
        signals.append("generations_improved")
    
    # Evidence synthesis
    generation_improved = "generations_improved" in signals
    loss_improved = "trajectory_improved" in signals or "perplexity_improved" in signals
    
    # Classify based on signals
    if generation_improved and loss_improved and len(signals) >= 3:
        status = "STRONGLY SUPPORTED"
        evidence = f"Multiple signals improved: {', '.join(signals)}"
    elif generation_improved or (loss_improved and len(signals) >= 2):
        status = "POSSIBLE"
        evidence = f"Some signals improved: {', '.join(signals)}"
    elif len(signals) == 1:
        status = "UNLIKELY"
        evidence = f"Minimal improvement: only {signals[0]}"
    elif len(signals) == 0:
        status = "RULED OUT"
        evidence = "No improvements from higher LR"
    else:
        status = "UNKNOWN"
        evidence = f"Mixed signals: {', '.join(signals)}"
    
    return status, evidence

def evaluate_h2_schedule_wrong(results: Dict[str, dict]) -> Tuple[str, str]:
    """H2: LR schedule wrong (cosine decay problematic)"""
    
    if "lr_003_constant" not in results or "lr_003_cosine" not in results:
        return "UNKNOWN", "Missing required experiments"
    
    constant = results["lr_003_constant"]
    cosine = results["lr_003_cosine"]
    
    signals = []
    
    # Compare trajectory
    traj_result = compare_trajectories(constant, cosine)
    if traj_result == "WORSE":  # Cosine worse means schedule issue
        signals.append("trajectory_worse_with_cosine")
    elif traj_result == "BETTER":
        signals.append("trajectory_better_with_cosine")
    
    # Compare perplexity
    ppl_result = compare_perplexity(constant, cosine)
    if ppl_result == "WORSE":
        signals.append("perplexity_worse_with_cosine")
    elif ppl_result == "BETTER":
        signals.append("perplexity_better_with_cosine")
    
    # Compare generations
    gen_result = compare_generations(constant, cosine)
    if gen_result == "WORSE":
        signals.append("generations_worse_with_cosine")
    elif gen_result == "BETTER":
        signals.append("generations_better_with_cosine")
    
    # Evidence synthesis
    worse_with_cosine = [s for s in signals if "worse" in s]
    better_with_cosine = [s for s in signals if "better" in s]
    
    if len(worse_with_cosine) >= 2 and len(better_with_cosine) == 0:
        status = "STRONGLY SUPPORTED"
        evidence = f"Cosine decay shows multiple problems: {', '.join(worse_with_cosine)}"
    elif len(worse_with_cosine) >= 1 and len(better_with_cosine) == 0:
        status = "POSSIBLE"
        evidence = f"Cosine decay shows: {', '.join(worse_with_cosine)}"
    elif len(better_with_cosine) >= 2:
        status = "RULED OUT"
        evidence = f"Cosine decay actually helps: {', '.join(better_with_cosine)}"
    elif len(signals) == 0:
        status = "UNLIKELY"
        evidence = "No difference between constant and cosine"
    else:
        status = "UNKNOWN"
        evidence = f"Mixed signals: {', '.join(signals)}"
    
    return status, evidence

# ============================================================
# Reporting Functions
# ============================================================

def print_experiment_summary(experiment: dict) -> None:
    """Print summary for single experiment"""
    
    print(f"\n{'=' * 70}")
    print(f"Experiment: {experiment['config_name']}")
    print(f"Description: {experiment['config']['description']}")
    print(f"{'=' * 70}")
    
    # Trajectory
    print(f"\nLoss Trajectory:")
    print(f"  Initial loss: {experiment['trajectory']['initial']:.4f}")
    print(f"  Final loss:   {experiment['trajectory']['final']:.4f}")
    print(f"  Best loss:    {experiment['trajectory']['best']:.4f}")
    print(f"  Reduction:    {experiment['trajectory']['reduction_pct']:.1f}%")
    print(f"  Still descending: {experiment['trajectory']['still_descending']}")
    
    # Perplexity
    print(f"\nPerplexity:")
    print(f"  Initial: {experiment['perplexity']['initial']:.1f}")
    print(f"  Final:   {experiment['perplexity']['final']:.1f}")
    print(f"  Best:    {experiment['perplexity']['best']:.1f}")
    print(f"  Reduction: {experiment['perplexity']['reduction_pct']:.1f}%")
    
    # Gradients
    print(f"\nGradient Statistics:")
    print(f"  Average norm: {experiment['gradients']['avg']:.6f}")
    print(f"  Min: {experiment['gradients']['min']:.6f}")
    print(f"  Max: {experiment['gradients']['max']:.6f}")
    print(f"  Healthy: {experiment['gradients']['healthy']}")
    
    # Generations
    print(f"\nGeneration Quality:")
    print(f"  Avg repetition score: {experiment['generations']['avg_repetition']:.1f}")
    print(f"  Coherent ratio: {experiment['generations']['coherent_ratio']:.1%}")
    print(f"  Coherent count: {experiment['generations']['coherent_count']}/{len(experiment['generations']['samples']) + 2}")
    
    print(f"\nSample Generation (first 100 chars):")
    for sample in experiment['generations']['samples'][:2]:
        print(f"  Prompt: {sample['prompt']}")
        print(f"    → {sample['generated']}")
        print(f"    Repetition: {sample['repetition']:.1f}")
    
    print()

def print_comparison_table(results: Dict[str, dict]) -> None:
    """Print comparison table across experiments"""
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    # Header
    print(f"\n{'Metric':<25}", end="")
    for name in results.keys():
        print(f"{name:<20}", end="")
    print()
    print("-" * 70)
    
    # Loss reduction
    print(f"{'Loss Reduction (%)':<25}", end="")
    for exp in results.values():
        print(f"{exp['trajectory']['reduction_pct']:<20.1f}", end="")
    print()
    
    # Best loss
    print(f"{'Best Loss':<25}", end="")
    for exp in results.values():
        print(f"{exp['trajectory']['best']:<20.4f}", end="")
    print()
    
    # PPL reduction
    print(f"{'PPL Reduction (%)':<25}", end="")
    for exp in results.values():
        print(f"{exp['perplexity']['reduction_pct']:<20.1f}", end="")
    print()
    
    # Avg gradient
    print(f"{'Avg Gradient Norm':<25}", end="")
    for exp in results.values():
        print(f"{exp['gradients']['avg']:<20.6f}", end="")
    print()
    
    # Avg repetition
    print(f"{'Avg Repetition Score':<25}", end="")
    for exp in results.values():
        print(f"{exp['generations']['avg_repetition']:<20.1f}", end="")
    print()
    
    # Coherent ratio
    print(f"{'Coherent Ratio':<25}", end="")
    for exp in results.values():
        print(f"{exp['generations']['coherent_ratio']:<20.1%}", end="")
    print()
    
    print()

def print_hypothesis_ranking(hypotheses: Dict[str, Tuple[str, str]]) -> None:
    """Print hypothesis ranking with evidence"""
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS RANKING")
    print("=" * 70)
    
    # Order by status severity
    status_order = {
        "STRONGLY SUPPORTED": 1,
        "POSSIBLE": 2,
        "UNKNOWN": 3,
        "UNLIKELY": 4,
        "RULED OUT": 5
    }
    
    sorted_hypotheses = sorted(
        hypotheses.items(),
        key=lambda x: status_order.get(x[1][0], 6)
    )
    
    for h_name, (status, evidence) in sorted_hypotheses:
        print(f"\n{h_name}:")
        print(f"  Status: [{status}]")
        print(f"  Evidence: {evidence}")
    
    print()

def compare_experiments(results_list):
    """Compare multiple experiments and rank hypotheses"""
    
    print("=" * 70)
    print("LR EXPERIMENT COMPARISON REPORT")
    print("=" * 70)
    
    # Extract signals for each experiment
    results = {}
    
    for result in results_list:
        exp_name = result["config_name"]
        exp_metrics = result["metrics_history"]
        
        # Extract all signals
        results[exp_name] = {
            "trajectory": extract_trajectory(exp_metrics),
            "perplexity": extract_perplexity(exp_metrics),
            "gradients": extract_gradient_stats(exp_metrics),
            "lr_behavior": extract_lr_behavior(exp_metrics),
            "generations": analyze_generations(result["generated_samples"]),
            "total_time": result.get("total_time_seconds", 0),
            "raw_results": result
        }
    
    # Print individual summaries
    for exp_name, exp_data in results.items():
        print_experiment_summary(exp_data)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Evaluate hypotheses
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    
    hypotheses = {}
    
    # H1: LR too low
    h1_status, h1_evidence = evaluate_h1_lr_too_low(results)
    hypotheses["H1_LR_TOO_LOW"] = (h1_status, h1_evidence)
    
    print(f"\nH1 (LR too low):")
    print(f"  Status: [{h1_status}]")
    print(f"  Evidence: {h1_evidence}")
    print(f"  Interpretation: Does higher LR (0.003 vs 0.001) improve learning?")
    
    # H2: Schedule wrong
    h2_status, h2_evidence = evaluate_h2_schedule_wrong(results)
    hypotheses["H2_SCHEDULE_WRONG"] = (h2_status, h2_evidence)
    
    print(f"\nH2 (Schedule wrong):")
    print(f"  Status: [{h2_status}]")
    print(f"  Evidence: {h2_evidence}")
    print(f"  Interpretation: Does cosine decay hurt vs constant?")
    
    # Additional hypotheses
    hypotheses["H3_MODEL_TOO_SMALL"] = ("UNKNOWN", "Requires architecture experiments")
    hypotheses["H4_TERNARY_LIMITATION"] = ("UNKNOWN", "Requires comparison with full-precision baseline")
    hypotheses["H5_NOT_ENOUGH_TRAINING"] = ("UNKNOWN", "Requires longer training run")
    hypotheses["H6_INIT_PROBLEM"] = ("UNKNOWN", "Static diagnostics show initialization is correct")
    
    # Print ranking
    print_hypothesis_ranking(hypotheses)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best experiment by generation quality
    best_by_generation = max(
        results.items(),
        key=lambda x: x[1]['generations']['coherent_ratio']
    )
    
    # Find best by loss
    best_by_loss = min(
        results.items(),
        key=lambda x: x[1]['trajectory']['best']
    )
    
    print(f"\nBest by generation quality: {best_by_generation[0]}")
    print(f"  Coherent ratio: {best_by_generation[1]['generations']['coherent_ratio']:.1%}")
    
    print(f"\nBest by loss: {best_by_loss[0]}")
    print(f"  Best loss: {best_by_loss[1]['trajectory']['best']:.4f}")
    
    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    # Check if any hypothesis is strongly supported
    strongly_supported = [h for h, (s, e) in hypotheses.items() if s == "STRONGLY SUPPORTED"]
    
    if strongly_supported:
        print(f"\nHypothesis/theses with strong support: {', '.join(strongly_supported)}")
        print("Evidence suggests tuning these parameters may help.")
    else:
        print("\nNo hypothesis has strong support from these experiments.")
        print("Issue may be elsewhere (architecture, data, or training duration).")
    
    # Check if ruled out
    ruled_out = [h for h, (s, e) in hypotheses.items() if s == "RULED OUT"]
    
    if ruled_out:
        print(f"\nHypothesis/theses ruled out: {', '.join(ruled_out)}")
        print("These are unlikely to be the root cause.")
    
    print("\n" + "=" * 70)
    
    return hypotheses

def main():
    parser = argparse.ArgumentParser(description="Compare LR experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Directory {results_dir} does not exist")
        sys.exit(1)
    
    # Find all experiment subdirectories
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not exp_dirs:
        print(f"ERROR: No experiment directories found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Load all results
    results_list = []
    for exp_dir in sorted(exp_dirs):
        result = load_experiment_results(exp_dir)
        if result:
            results_list.append(result)
            print(f"Loaded: {exp_dir.name}")
    
    if not results_list:
        print("ERROR: No valid results found")
        sys.exit(1)
    
    # Compare
    compare_experiments(results_list)

if __name__ == "__main__":
    main()
