# MLOps Project 2 — Containerization

This project packages a PyTorch Lightning training pipeline for **DistilBERT** on **GLUE (MRPC)** into a **Dockerized CLI app**.  
Goal: **single-command reproducibility** across machines.

---

## Quickstart (Docker — recommended)

No local Python setup needed. Everything (dependencies, environment variables) is in the image.

```bash
# Build the image
docker build -t mlops-cont:cpu .

# Run training (3 epochs by default)
docker run --rm -v "$(pwd)/models:/app/models" mlops-cont:cpu
```

Override hyperparameters (example: 3 epochs, custom lr/weight-decay/warmup_ratio)

```bash
docker run --rm -v "$(pwd)/models:/app/models" mlops-cont:cpu \
  --epochs 3 --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.06
```

## If you prefer to run without Docker (hyperparameters also adjustable):

```bash
pip install -r requirements.txt
python main.py --checkpoint_dir models --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.06 --epochs 3
```

## Main Arguments one can choose:

--checkpoint_dir (str) — where to save checkpoints (default: models)

--lr (float) — learning rate (default: 2e-5)

--weight_decay (float) — weight decay (default: 0.0)

--warmup_ratio (float) — ratio of warmup steps (default: 0.0)

--epochs (int) — number of epochs (default: 3)