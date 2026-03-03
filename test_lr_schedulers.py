import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def main():
    plt.xkcd()
    font_family = None
    for f in font_manager.fontManager.ttflist:
        if "xkcd" in f.name.lower():
            font_family = f.name
            break
    if font_family is not None:
        plt.rcParams['font.family'] = font_family

    peak_lr = 0.0014
    warmup_period = 1600
    decay_period = 32000

    model = nn.Sequential(
        nn.Linear(34, 34),
        nn.ReLU(),
        nn.Linear(34, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr, weight_decay=0.1
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.00001,
        end_factor=1.0,
        total_iters=warmup_period
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=decay_period,
        eta_min=peak_lr / 10
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_period],
    )

    lrs = []
    for ii in range(warmup_period + decay_period):
        if ii == 0:
            print("Initial learning rate: ", optimizer.param_groups[0]["lr"])
        elif warmup_period - 5 < ii < warmup_period + 5:
            print(f"Step {ii} learning rate: ", optimizer.param_groups[0]["lr"])
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    steps = np.arange(warmup_period + decay_period)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(steps, lrs, 'b-', linewidth=2)
    ax.set_xlim(0, warmup_period + decay_period)
    ax.set_ylim(0, peak_lr * 1.1)
    ax.set_xlabel('STEP')
    ax.set_ylabel('LEARNING RATE')

    fig.tight_layout()
    fig.savefig('plot_actual_schedule.png', bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":
    main()
