#!/bin/bash
# Demo: Run single experiment for each dataset with seed 42

echo "╔════════════════════════════════════════════════╗"
echo "║  Quick Demo: HuggingFace Datasets + Seeds     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "This will run:"
echo "  - iSarcasm dataset with seed 42 (5 epochs)"
echo "  - SemEval 3A dataset with seed 42 (5 epochs)"
echo ""

# Activate virtual environment
source venv/bin/activate

# iSarcasm
echo "▶ Running iSarcasm experiment..."
python3 prepare_hf_dataset.py --dataset isarcasm
python3 build_graph.py isarcasm --seed 42
python3 train_bert_gcn.py --dataset isarcasm --seed 42 --device cpu --nb_epochs 5 --batch_size 32

# SemEval 3A
echo ""
echo "▶ Running SemEval 3A experiment..."
python3 prepare_hf_dataset.py --dataset semeval3a
python3 build_graph.py semeval3a --seed 42
python3 train_bert_gcn.py --dataset semeval3a --seed 42 --device cpu --nb_epochs 5 --batch_size 32

echo ""
echo "✓ Demo complete! Check ./checkpoint/ for results."
