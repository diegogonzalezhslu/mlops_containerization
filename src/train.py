from typing import Optional, List
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.glue_datamodule import GLUEDataModule
from src.glue_model import GLUETransformer

def make_wandb_logger(base_name, project="mlops-hparam-mrpc", tags=None):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return WandbLogger(
        project=project,
        name=f"{base_name}_{ts}",
        tags=tags or [],
        reinit=True,
        resume="never",
    )

def run_experiment(
    *,
    model_name="distilbert-base-uncased",
    task_name="mrpc",
    train_bs=16,
    eval_bs=32,
    lr=2e-5,
    weight_decay=0.0,
    warmup_steps=0,
    warmup_ratio=0.0,
    accumulate_grad_batches=1,
    seed=42,
    base_run_name="exp",
    tags=None,
    checkpoint_dir="models",
    max_epochs=3,
    project_name="mlops-hparam-mrpc",
):
    # make deterministic(ish)
    L.seed_everything(seed)

    # datamodule
    dm = GLUEDataModule(
        model_name_or_path=model_name,
        task_name=task_name,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
    )
    dm.setup("fit")

    # model
    model = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
    )

    # W&B logger (uses your exact pattern)
    wandb_logger = make_wandb_logger(
        base_name=f"{base_run_name}_lr{lr}_wd{weight_decay}_wu{warmup_ratio}_acc{accumulate_grad_batches}",
        project=project_name,
        tags=tags,
    )
    # store config
    wandb_logger.experiment.config.update({
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "train_bs": train_bs,
        "eval_bs": eval_bs,
        "accumulate_grad_batches": accumulate_grad_batches,
        "seed": seed,
        "task_name": task_name,
        "model_name": model_name,
        "epochs": max_epochs,
    })

    # callbacks: LR monitor + checkpoint to checkpoint_dir
    lr_cb = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="glue-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,         # defaults to 3 (matches your notebook)
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[lr_cb, ckpt_cb],
    )
    trainer.fit(model, datamodule=dm)

    # Ensure W&B ends cleanly
    import wandb
    wandb.finish()
