# BertGCN with HuggingFace Datasets Support

## Updated Features

### 1. HuggingFace Dataset Support
Now supports loading datasets directly from HuggingFace:
- `viethq1906/isarcasm_2022_taskA_En` → dataset name: `isarcasm`
- `viethq1906/semeval_2018_3A` → dataset name: `semeval3a`

### 2. Reproducibility with Seeds
Full reproducibility support with random seeds (42-46):
- Sets seeds for: random, numpy, torch, torch.cuda, DGL
- Deterministic CUDA operations enabled

### 3. CPU/GPU Device Selection
Flexible device selection with `--device` flag

## Quick Start

### Prepare HuggingFace Datasets
```bash
# Prepare iSarcasm dataset
python3 prepare_hf_dataset.py --dataset isarcasm

# Prepare SemEval 3A dataset
python3 prepare_hf_dataset.py --dataset semeval3a

# Prepare both
python3 prepare_hf_dataset.py --dataset all
```

### Build Graph with Seed
```bash
# Build graph for iSarcasm with seed 42
python3 build_graph.py isarcasm --seed 42

# Build graph for SemEval 3A with seed 43
python3 build_graph.py semeval3a --seed 43
```

### Train Model
```bash
# Train on iSarcasm with seed 42 on CPU
python3 train_bert_gcn.py --dataset isarcasm --seed 42 --device cpu --nb_epochs 50

# Train on SemEval 3A with seed 43 on GPU
python3 train_bert_gcn.py --dataset semeval3a --seed 43 --device cuda --nb_epochs 50
```

### Run Complete Experiments
```bash
# Run all experiments with seeds 42-46 on both datasets
python3 run_experiments.py --datasets isarcasm semeval3a --seeds 42 43 44 45 46 --nb_epochs 50 --device cpu

# Or use the bash script
chmod +x run_hf_experiments.sh
./run_hf_experiments.sh
```

## Command Line Arguments

### prepare_hf_dataset.py
- `--dataset`: Dataset to prepare (`isarcasm`, `semeval3a`, or `all`)
- `--output_dir`: Output directory (default: `data`)

### build_graph.py
- `dataset`: Dataset name (positional argument)
- `--seed`: Random seed for reproducibility (default: 42)

### train_bert_gcn.py (New arguments)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--dataset`: Now includes `isarcasm` and `semeval3a`

### run_experiments.py
- `--datasets`: List of datasets to run (default: `isarcasm semeval3a`)
- `--seeds`: List of random seeds (default: `42 43 44 45 46`)
- `--gcn_model`: GCN model type (`gcn` or `gat`, default: `gcn`)
- `--nb_epochs`: Number of training epochs (default: 50)
- `--device`: Device to use (`cuda` or `cpu`, default: `cpu`)

## Examples

### Single experiment
```bash
# Prepare dataset
python3 prepare_hf_dataset.py --dataset isarcasm

# Build graph with seed 42
python3 build_graph.py isarcasm --seed 42

# Train model
python3 train_bert_gcn.py --dataset isarcasm --seed 42 --nb_epochs 50 --device cpu
```

### Multiple seeds for reproducibility
```bash
for seed in 42 43 44 45 46
do
    python3 build_graph.py isarcasm --seed $seed
    python3 train_bert_gcn.py --dataset isarcasm --seed $seed --nb_epochs 50 --device cpu
done
```

### Automated experiment runner
```bash
# Run experiments on both datasets with 5 seeds
python3 run_experiments.py \
    --datasets isarcasm semeval3a \
    --seeds 42 43 44 45 46 \
    --nb_epochs 50 \
    --device cpu \
    --gcn_model gcn
```

## Output Structure

Results are saved in:
```
./checkpoint/{dataset}_seed{seed}_{gcn_model}_{timestamp}/
├── training.log        # Training logs
├── train_bert_gcn.py   # Copy of training script
└── model_*.pkl         # Saved model checkpoints
```

## Notes

1. **First run**: The script will automatically download datasets from HuggingFace
2. **Reproducibility**: Using the same seed ensures identical results across runs
3. **GPU compatibility**: If GPU is not compatible (like Tesla M40 with CUDA 12), use `--device cpu`
4. **Progress bars**: All long-running operations now have tqdm progress bars
