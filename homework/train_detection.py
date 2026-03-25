import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data


from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .metrics import DetectionMetric  # Provided

import torch.nn.functional as F

def train(
    exp_dir: str = "detector_logs",
    model_name: str = "detector",
    l = 1.0, # lambda for depth loss
    lr: float = 0.001,
    num_epoch: int = 50,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else
                          "cpu")
    print(f"Using device: {device}")

    # deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # tensorboard log directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    # model
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    # data loaders
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    # loss functions
    seg_loss_fn = torch.nn.CrossEntropyLoss()
    depth_loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    train_perf = DetectionMetric()
    val_perf = DetectionMetric()
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        depth_mae_total = 0.0
        lane_mae_total = 0.0
        count = 0
        train_perf.reset()

        for data_dict in train_data:
            img = data_dict['image'].to(device)
            seg_labels = data_dict['track'].to(device)
            depth_target = data_dict['depth'].to(device)

            optimizer.zero_grad()
            seg_logits, depth_pred = model(img)
            depth_pred = depth_pred.squeeze(1)
            
            # combined loss
            loss = seg_loss_fn(seg_logits, seg_labels) + l * depth_loss_fn(depth_pred, depth_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)

            # update metrics
            pred_labels = torch.argmax(seg_logits, dim=1)
            train_perf.add(pred_labels, seg_labels,
                            depth_pred,depth_target)

            lane_mask = (seg_labels > 0)
            if lane_mask.any():
                lane_mae_total += torch.abs(depth_pred - depth_target)[lane_mask].sum().item()
            depth_mae_total += torch.abs(depth_pred - depth_target).sum().item()
            count += img.size(0) * img.shape[2] * img.shape[3]  # total pixels

            global_step += 1

        # epoch metrics
        train_dict = train_perf.compute()
        train_acc = train_dict['accuracy']
        train_IOU = train_dict['iou']
        train_MAE = train_dict['abs_depth_error']
        train_lane_MAE = train_dict['tp_depth_error']

        logger.add_scalar('train_mIoU', train_IOU, epoch)
        logger.add_scalar('train_depth_MAE', train_MAE, epoch)
        logger.add_scalar('train_lane_MAE',train_lane_MAE)
        logger.add_scalar('train_accuracy', train_acc, epoch)

        # validation
        model.eval()
        val_loss = 0.0
        val_perf.reset()
        val_depth_mae_total = 0.0
        val_lane_mae_total = 0.0
        val_count = 0

        with torch.inference_mode():
            for data_dict in val_data:
                img = data_dict['image'].to(device)
                seg_labels = data_dict['track'].to(device)
                depth_target = data_dict['depth'].to(device)

                seg_logits, depth_pred = model(img)
                depth_pred = depth_pred.squeeze(1)
                loss = seg_loss_fn(seg_logits, seg_labels) + l * depth_loss_fn(depth_pred, depth_target)
                val_loss += loss.item() * img.size(0)

                pred_labels = torch.argmax(seg_logits, dim=1)
                val_perf.add(pred_labels, seg_labels,
                            depth_pred,depth_target)

                lane_mask = (seg_labels > 0)
                if lane_mask.any():
                    val_lane_mae_total += torch.abs(depth_pred - depth_target)[lane_mask].sum().item()
                val_depth_mae_total += torch.abs(depth_pred - depth_target).sum().item()
                val_count += img.size(0) * img.shape[2] * img.shape[3]

        val_dict = val_perf.compute()
        val_acc = val_dict['accuracy']
        val_IOU = val_dict['iou']
        val_MAE = val_dict['abs_depth_error']
        val_lane_MAE = val_dict['tp_depth_error']

        logger.add_scalar('val_mIoU', val_IOU, epoch)
        logger.add_scalar('val_depth_MAE', val_MAE, epoch)
        logger.add_scalar('val_lane_MAE',val_lane_MAE)
        logger.add_scalar('val_accuracy', val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % (num_epoch/10) == 0:
          print(f"Epoch {epoch+1:2d}/{num_epoch:2d} | "
              f"Train - Acc: {train_acc:.2f} | IoU: {train_IOU:.3f} | Depth MAE: {train_MAE:.3f} | Lane Acc: {train_lane_MAE:.3f} || ",
              f"Val --- Acc: {val_acc:.2f} | IoU: {val_IOU:.3f} | Depth MAE: {val_MAE:.3f} | Lane Acc: {val_lane_MAE:.3f} || ")

    # save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_lyrs", type=int,nargs="+",
     default=[64,64,64,64])
    

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
