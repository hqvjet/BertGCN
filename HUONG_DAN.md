# Hướng Dẫn Sử Dụng BertGCN với HuggingFace Datasets

## Tính Năng Mới

✅ **Hỗ trợ 2 datasets từ HuggingFace:**
- `viethq1906/isarcasm_2022_taskA_En` (dataset: `isarcasm`)
- `viethq1906/semeval_2018_3A` (dataset: `semeval3a`)

✅ **Reproducibility với seeds 42-46:**
- Cài đặt seeds cho: random, numpy, torch, cuda, DGL
- Đảm bảo kết quả giống nhau giữa các lần chạy

✅ **Chọn device CPU/GPU:**
- `--device cpu`: Chạy trên CPU (an toàn cho mọi máy)
- `--device cuda`: Chạy trên GPU (nếu tương thích)

## Cách Sử Dụng

### 1. Chạy Demo Nhanh (Khuyến nghị)
```bash
# Demo với 1 seed, mỗi dataset 5 epochs
./demo_quick.sh
```

### 2. Chạy Thủ Công (Step-by-step)

#### Bước 1: Chuẩn bị dataset
```bash
# iSarcasm
python3 prepare_hf_dataset.py --dataset isarcasm

# SemEval 3A
python3 prepare_hf_dataset.py --dataset semeval3a

# Hoặc cả hai
python3 prepare_hf_dataset.py --dataset all
```

#### Bước 2: Build graph với seed
```bash
# iSarcasm với seed 42
python3 build_graph.py isarcasm --seed 42

# SemEval 3A với seed 43
python3 build_graph.py semeval3a --seed 43
```

#### Bước 3: Train model
```bash
# Train trên CPU (an toàn)
python3 train_bert_gcn.py \
  --dataset isarcasm \
  --seed 42 \
  --device cpu \
  --nb_epochs 50

# Train trên GPU (nếu tương thích)
python3 train_bert_gcn.py \
  --dataset semeval3a \
  --seed 42 \
  --device cuda \
  --nb_epochs 50
```

### 3. Chạy Nhiều Experiments với Seeds 42-46

#### Cách 1: Sử dụng script Python
```bash
python3 run_experiments.py \
  --datasets isarcasm semeval3a \
  --seeds 42 43 44 45 46 \
  --nb_epochs 50 \
  --device cpu
```

#### Cách 2: Sử dụng bash script
```bash
./run_hf_experiments.sh
```

### 4. Test Tích Hợp
```bash
# Test xem mọi thứ có hoạt động không
python3 test_hf_integration.py
```

## Tham Số Quan Trọng

### prepare_hf_dataset.py
- `--dataset`: Dataset cần chuẩn bị (`isarcasm`, `semeval3a`, hoặc `all`)
- `--output_dir`: Thư mục output (mặc định: `data`)

### build_graph.py
- `dataset`: Tên dataset (bắt buộc)
- `--seed`: Random seed (mặc định: 42)

### train_bert_gcn.py
- `--dataset`: Dataset để train
- `--seed`: Random seed (mặc định: 42)
- `--device`: `cpu` hoặc `cuda` (mặc định: `cuda`)
- `--nb_epochs`: Số epochs (mặc định: 50)
- `--batch_size`: Batch size (mặc định: 64)
- `--bert_init`: BERT model (`roberta-base`, `bert-base-uncased`, etc.)

### run_experiments.py
- `--datasets`: Danh sách datasets (mặc định: `isarcasm semeval3a`)
- `--seeds`: Danh sách seeds (mặc định: `42 43 44 45 46`)
- `--nb_epochs`: Số epochs (mặc định: 50)
- `--device`: Device (`cpu` hoặc `cuda`)
- `--gcn_model`: Model type (`gcn` hoặc `gat`)

## Ví Dụ

### Chạy 1 experiment
```bash
# Chuẩn bị
python3 prepare_hf_dataset.py --dataset isarcasm

# Build graph
python3 build_graph.py isarcasm --seed 42

# Train
python3 train_bert_gcn.py \
  --dataset isarcasm \
  --seed 42 \
  --nb_epochs 50 \
  --device cpu \
  --batch_size 32
```

### Chạy với nhiều seeds (reproducibility)
```bash
for seed in 42 43 44 45 46
do
  echo "Running with seed $seed..."
  python3 build_graph.py isarcasm --seed $seed
  python3 train_bert_gcn.py \
    --dataset isarcasm \
    --seed $seed \
    --nb_epochs 50 \
    --device cpu
done
```

### Chạy tất cả (2 datasets × 5 seeds = 10 experiments)
```bash
python3 run_experiments.py \
  --datasets isarcasm semeval3a \
  --seeds 42 43 44 45 46 \
  --nb_epochs 50 \
  --device cpu \
  --gcn_model gcn
```

## Kết Quả

Kết quả được lưu trong:
```
./checkpoint/{dataset}_seed{seed}_{gcn_model}_{timestamp}/
├── training.log        # Log chi tiết
├── train_bert_gcn.py   # Copy của script
└── model_*.pkl         # Model checkpoints
```

## Lưu Ý

1. **Lần đầu chạy**: Script sẽ tự động download datasets từ HuggingFace
2. **Reproducibility**: Dùng cùng seed để có kết quả giống nhau
3. **GPU không tương thích**: Nếu GPU không support (như Tesla M40 với CUDA 12), dùng `--device cpu`
4. **Progress bars**: Tất cả operations dài đều có progress bars (tqdm)
5. **Batch size**: Giảm batch size nếu gặp lỗi out of memory

## Troubleshooting

### GPU không tương thích
```bash
# Dùng CPU thay vì GPU
python3 train_bert_gcn.py --dataset isarcasm --device cpu
```

### Out of Memory
```bash
# Giảm batch size
python3 train_bert_gcn.py --dataset isarcasm --batch_size 16 --device cpu
```

### Import Error
```bash
# Cài đặt lại dependencies
pip install -r requirements.txt
```

## Support

Các datasets được hỗ trợ:
- `20ng`, `R8`, `R52`, `ohsumed`, `mr` (datasets gốc)
- `isarcasm` (viethq1906/isarcasm_2022_taskA_En)
- `semeval3a` (viethq1906/semeval_2018_3A)

Random seeds khuyến nghị: **42, 43, 44, 45, 46**

Device options: **cpu** (an toàn) hoặc **cuda** (nếu tương thích)
