#!/bin/bash
# Quick run script for new HuggingFace datasets with seeds 42-46

echo "=========================================="
echo "Running experiments on HuggingFace datasets"
echo "Datasets: isarcasm, semeval3a"
echo "Seeds: 42, 43, 44, 45, 46"
echo "=========================================="

# Option 1: Run all experiments automatically
# python3 run_experiments.py --datasets isarcasm semeval3a --seeds 42 43 44 45 46 --nb_epochs 10 --device cpu

# Option 2: Run individual experiments (uncomment to use)

# iSarcasm dataset
echo "Preparing iSarcasm dataset..."
python3 prepare_hf_dataset.py --dataset isarcasm

for seed in 42 43 44 45 46
do
    echo "Building graph for iSarcasm with seed $seed..."
    python3 build_graph.py isarcasm --seed $seed
    
    echo "Training iSarcasm with seed $seed..."
    python3 train_bert_gcn.py --dataset isarcasm --seed $seed --nb_epochs 10 --device cpu
done

# SemEval dataset
echo "Preparing SemEval 3A dataset..."
python3 prepare_hf_dataset.py --dataset semeval3a

for seed in 42 43 44 45 46
do
    echo "Building graph for semeval3a with seed $seed..."
    python3 build_graph.py semeval3a --seed $seed
    
    echo "Training semeval3a with seed $seed..."
    python3 train_bert_gcn.py --dataset semeval3a --seed $seed --nb_epochs 10 --device cpu
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
