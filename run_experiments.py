"""
Run reproducible experiments with multiple seeds
"""
import os
import subprocess
import argparse
from datetime import datetime

def run_experiment(dataset, seed, gcn_model='gcn', nb_epochs=50, device='cpu'):
    """Run a single experiment with specified seed"""
    
    print(f"\n{'='*80}")
    print(f"Running experiment: dataset={dataset}, seed={seed}, gcn_model={gcn_model}")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare HuggingFace dataset if needed
    if dataset in ['isarcasm', 'semeval3a']:
        print(f"Preparing {dataset} dataset...")
        result = subprocess.run(
            ['python3', 'prepare_hf_dataset.py', '--dataset', dataset],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"✗ Failed to prepare dataset {dataset}")
            return False
    
    # Step 2: Build graph with seed
    print(f"\nBuilding graph for {dataset} with seed {seed}...")
    result = subprocess.run(
        ['python3', 'build_graph.py', dataset, '--seed', str(seed)],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"✗ Failed to build graph for {dataset}")
        return False
    
    # Step 3: Train model with seed
    checkpoint_dir = f'./checkpoint/{dataset}_seed{seed}_{gcn_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f"\nTraining model with seed {seed}...")
    cmd = [
        'python3', 'train_bert_gcn.py',
        '--dataset', dataset,
        '--seed', str(seed),
        '--nb_epochs', str(nb_epochs),
        '--gcn_model', gcn_model,
        '--checkpoint_dir', checkpoint_dir,
        '--device', device
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"✗ Failed to train model for {dataset} with seed {seed}")
        return False
    
    print(f"\n✓ Experiment completed: {dataset}, seed {seed}")
    print(f"  Results saved to: {checkpoint_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run reproducible experiments with multiple seeds')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['isarcasm', 'semeval3a'],
                       choices=['isarcasm', 'semeval3a', '20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                       help='Datasets to run experiments on')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 43, 44, 45, 46],
                       help='Random seeds for reproducibility')
    parser.add_argument('--gcn_model', type=str, default='gcn',
                       choices=['gcn', 'gat'],
                       help='GCN model type')
    parser.add_argument('--nb_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("REPRODUCIBLE EXPERIMENT RUNNER")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"GCN Model: {args.gcn_model}")
    print(f"Epochs: {args.nb_epochs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            success = run_experiment(
                dataset=dataset,
                seed=seed,
                gcn_model=args.gcn_model,
                nb_epochs=args.nb_epochs,
                device=args.device
            )
            results.append({
                'dataset': dataset,
                'seed': seed,
                'success': success
            })
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} Dataset: {result['dataset']}, Seed: {result['seed']}")
    
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal: {successful}/{total} experiments successful")
    print("="*80)

if __name__ == '__main__':
    main()
