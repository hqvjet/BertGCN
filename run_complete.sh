#!/bin/bash
# Complete workflow: Prepare → Build → Train for both datasets with seed 42

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  BertGCN - Complete Pipeline for HuggingFace Datasets        ║"
echo "║  Datasets: iSarcasm + SemEval 3A                             ║"
echo "║  Seed: 42 (reproducible)                                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found. Please create one first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 1/6: Prepare iSarcasm Dataset"
echo "═══════════════════════════════════════════════════════════════"
python3 prepare_hf_dataset.py --dataset isarcasm

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 2/6: Build Graph for iSarcasm (seed=42)"
echo "═══════════════════════════════════════════════════════════════"
python3 build_graph.py isarcasm --seed 42

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 3/6: Train Model on iSarcasm (seed=42, 10 epochs)"
echo "═══════════════════════════════════════════════════════════════"
python3 train_bert_gcn.py \
    --dataset isarcasm \
    --seed 42 \
    --device cpu \
    --nb_epochs 10 \
    --batch_size 32

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 4/6: Prepare SemEval 3A Dataset"
echo "═══════════════════════════════════════════════════════════════"
python3 prepare_hf_dataset.py --dataset semeval3a

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 5/6: Build Graph for SemEval 3A (seed=42)"
echo "═══════════════════════════════════════════════════════════════"
python3 build_graph.py semeval3a --seed 42

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 6/6: Train Model on SemEval 3A (seed=42, 10 epochs)"
echo "═══════════════════════════════════════════════════════════════"
python3 train_bert_gcn.py \
    --dataset semeval3a \
    --seed 42 \
    --device cpu \
    --nb_epochs 10 \
    --batch_size 32

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✓ COMPLETE! All experiments finished successfully           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: ./checkpoint/"
echo ""
echo "To run with more seeds (42-46), use:"
echo "  python3 run_experiments.py --datasets isarcasm semeval3a --seeds 42 43 44 45 46"
