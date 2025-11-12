# MLOps Project 2 – Containerization

This project containerizes a PyTorch Lightning training pipeline for GLUE (MRPC) fine-tuning using the DistilBERT model.

## 💡 Run locally

```bash
pip install -r requirements.txt
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
python main.py --checkpoint_dir models --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.06
