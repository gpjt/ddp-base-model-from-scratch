import datetime
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch


def get_checkpoints_dir(run_dir):
    return run_dir / "checkpoints"


def load_checkpoint(run_dir, checkpoint, model, optimizer=None, scaler=None):
    checkpoints_dir = get_checkpoints_dir(run_dir)
    checkpoint_dir = checkpoints_dir / checkpoint
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))

    if optimizer:
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))

    if scaler:
        scaler.load_state_dict(torch.load(checkpoint_dir / "scaler.pt"))

    with open(checkpoint_dir / "meta.json", "r") as f:
        meta = json.load(f)
        restart_global_step = meta["global_step"] + 1

    with open(checkpoints_dir / "best" / "meta.json") as f:
        best_loss = json.load(f)["avg_train_loss"]

    return restart_global_step, best_loss


def save_checkpoint(
    run_dir,
    name,
    model, optimizer, scaler,
    min_train_loss, max_train_loss, avg_train_loss,
    max_grad_norms, avg_grad_norms, frac_clipped,
    global_step, is_best
):
    checkpoints_dir = get_checkpoints_dir(run_dir)
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir()

    now = datetime.datetime.now(datetime.UTC)
    checkpoint_name = f"{now:%Y%m%dZ%H%M%S}-{name}"
    checkpoint_dir = checkpoints_dir / checkpoint_name
    checkpoint_dir.mkdir()

    save_file(model.state_dict(), checkpoint_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")

    with open(checkpoint_dir / "meta.json", "w") as f:
        json.dump(
            dict(
                min_train_loss=min_train_loss, 
                max_train_loss=max_train_loss,
                avg_train_loss=avg_train_loss,
                max_grad_norms=max_grad_norms, 
                avg_grad_norms=avg_grad_norms, 
                frac_clipped=frac_clipped,
                global_step=global_step,
                is_best=is_best,
            ),
            f
        )

    symlink_target = Path(".") / checkpoint_dir.name
    if is_best:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = checkpoints_dir / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)
