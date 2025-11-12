import argparse
import os
from src.train import run_experiment

def comma_split(v):
    if v is None or v == "":
        return None
    return [x.strip() for x in str(v).split(",") if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description="Single-call GLUE training (Lightning + HF + W&B)")
    # core
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--task_name", type=str, default="mrpc")
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # logging + run naming
    p.add_argument("--project_name", type=str, default="mlops-hparam-mrpc")
    p.add_argument("--run_base_name", type=str, default="exp")
    p.add_argument("--tags", type=str, default="containerized,task1")

    # checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="models")

    # epochs (defaults to 3 like your notebook; you can override if allowed)
    p.add_argument("--epochs", type=int, default=3)

    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # normalize tags
    tag_list = comma_split(args.tags)

    run_experiment(
        model_name=args.model_name,
        task_name=args.task_name,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        accumulate_grad_batches=args.accumulate_grad_batches,
        seed=args.seed,
        base_run_name=args.run_base_name,
        tags=tag_list,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.epochs,
        project_name=args.project_name,
    )

if __name__ == "__main__":
    main()
