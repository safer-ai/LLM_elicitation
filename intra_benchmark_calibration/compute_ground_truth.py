import json
import numpy as np
from pathlib import Path
import argparse

def compute_conditional_probabilities(json_path: str, n_bins: int = 4,
                                     min_sample_size: int = 5):
    """
    Compute conditional probabilities P(j|i) for intra-benchmark calibration.

    Parameters:
    -----------
    json_path : str
        Path to JSON file with model scores
    n_bins : int
        Number of bins to divide the score range into
    min_sample_size : int
        Minimum sample size to compute ground truth (default: 5)

    Returns:
    --------
    dict : Contains ground truth probabilities, bin info, and metadata
    """
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scores = np.array([m['score'] for m in data['results']])
    model_names = [m['model'] for m in data['results']]
    n_models = len(scores)
    
    # Create bins
    min_score = scores.min()
    max_score = scores.max()
    bin_edges = np.linspace(min_score, max_score + 0.01, n_bins + 1)
    
    # Assign models to bins
    model_bins = np.digitize(scores, bin_edges) - 1
    model_bins = np.clip(model_bins, 0, n_bins - 1)
    
    # Count models reaching each bin threshold
    # A model "reaches bin i" if it's in bin i or higher
    models_reaching_bin = {}
    for i in range(n_bins):
        models_reaching_bin[i] = np.sum(model_bins >= i)
    
    # Compute all conditional probabilities P(j|i)
    results = []
    for i in range(n_bins):
        for j in range(i + 1, n_bins):  # j > i only (monotonic progression)
            n_reaching_i = models_reaching_bin[i]
            n_reaching_j = models_reaching_bin[j]
            
            if n_reaching_i >= min_sample_size:
                p_j_given_i = n_reaching_j / n_reaching_i if n_reaching_i > 0 else np.nan
                sufficient_sample = True
            else:
                p_j_given_i = np.nan
                sufficient_sample = False
            
            results.append({
                'bin_i': i,
                'bin_j': j,
                'bin_i_range': f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
                'bin_j_range': f"[{bin_edges[j]:.1f}, {bin_edges[j+1]:.1f})",
                'n_reaching_i': int(n_reaching_i),
                'n_reaching_j': int(n_reaching_j),
                'p_j_given_i': p_j_given_i,
                'sufficient_sample': sufficient_sample
            })
    
    # Create bin metadata
    bin_metadata = []
    for i in range(n_bins):
        models_in_bin = [model_names[k] for k in range(n_models) if model_bins[k] == i]
        bin_metadata.append({
            'bin_id': i,
            'range': f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
            'n_models_in_bin': len(models_in_bin),
            'n_models_reaching_bin': int(models_reaching_bin[i]),
            'models': models_in_bin
        })
    
    # Compile output
    output = {
        'metadata': {
            'n_bins': n_bins,
            'n_models': n_models,
            'score_range': [float(min_score), float(max_score)],
            'min_sample_size': min_sample_size,
            'total_predictions': len(results),
            'sufficient_sample_count': sum(r['sufficient_sample'] for r in results)
        },
        'bin_info': bin_metadata,
        'ground_truth': results
    }

    # Print summary
    print("\n=== Summary ===")
    print(f"Total models: {n_models}")
    print(f"Number of bins: {n_bins}")
    print(f"Total predictions: {len(results)}")
    print(f"Predictions with sufficient sample (n≥{min_sample_size}): "
          f"{sum(r['sufficient_sample'] for r in results)}")
    print("\n=== Bin Distribution ===")
    for bm in bin_metadata:
        print(f"Bin {bm['bin_id']} {bm['range']}: "
              f"{bm['n_models_in_bin']} models in bin, "
              f"{bm['n_models_reaching_bin']} reaching this threshold")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute intra-benchmark ground truth probabilities')
    parser.add_argument('-i', '--input-path', type=str,
                        default='input_data/ground_truth/cybench_leaderboard.json',
                        help='Path to input JSON file with model scores')
    parser.add_argument('-o', '--output-path', type=str, default=None,
                        help='Path to output file (default: input path with _leaderboard.json replaced by _ground_truth_n{n}.json)')
    parser.add_argument('--bin-count', type=int, default=4,
                        help='Number of bins to divide the score range into (default: 4)')

    args = parser.parse_args()

    BIN_COUNT = args.bin_count

    # Set default output path if not specified
    if args.output_path is None:
        input_path = Path(args.input_path)
        output_filename = input_path.name.replace('_leaderboard.json', f'_ground_truth_n{BIN_COUNT}.json')
        output_path = input_path.parent / output_filename
    else:
        output_path = Path(args.output_path)

    print(f"\n{'='*60}")
    print(f"Computing with {BIN_COUNT} bins")
    print('='*60)
    result = compute_conditional_probabilities(
        args.input_path,
        n_bins=BIN_COUNT,
        min_sample_size=5
    )

    # Save JSON output to specified path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved JSON output to: {output_path}")
