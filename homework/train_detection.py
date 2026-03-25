import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .utils import load_data


from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .datasets.road_dataset import ConfusionMatrix  # Provided

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

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        conf_matrix = ConfusionMatrix(num_classes=3)  # segmentation classes
        depth_mae_total = 0.0
        lane_mae_total = 0.0
        count = 0

        for img, label in train_data:
            img = img.to(device)
            seg_labels = label[0].to(device)   # (B,H,W)
            depth_target = label[1].to(device) # (B,H,W)

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
            conf_matrix.update(pred_labels.cpu().numpy(), seg_labels.cpu().numpy())

            lane_mask = (seg_labels > 0)
            if lane_mask.any():
                lane_mae_total += torch.abs(depth_pred - depth_target)[lane_mask].sum().item()
            depth_mae_total += torch.abs(depth_pred - depth_target).sum().item()
            count += img.size(0) * img.shape[2] * img.shape[3]  # total pixels

            global_step += 1

        # epoch metrics
        epoch_loss = train_loss / len(train_data.dataset)
        train_miou = conf_matrix.iou().mean()
        train_depth_mae = depth_mae_total / count
        train_lane_mae = lane_mae_total / count

        logger.add_scalar('train_loss', epoch_loss, epoch)
        logger.add_scalar('train_mIoU', train_miou, epoch)
        logger.add_scalar('train_depth_MAE', train_depth_mae, epoch)
        logger.add_scalar('train_lane_MAE', train_lane_mae, epoch)

        # validation
        model.eval()
        val_loss = 0.0
        val_conf_matrix = ConfusionMatrix(num_classes=3)
        val_depth_mae_total = 0.0
        val_lane_mae_total = 0.0
        val_count = 0

        with torch.inference_mode():
            for img, label in val_data:
                img = img.to(device)
                seg_labels = label[0].to(device)
                depth_target = label[1].to(device)

                seg_logits, depth_pred = model(img)
                depth_pred = depth_pred.squeeze(1)
                loss = seg_loss_fn(seg_logits, seg_labels) + l * depth_loss_fn(depth_pred, depth_target)
                val_loss += loss.item() * img.size(0)

                pred_labels = torch.argmax(seg_logits, dim=1)
                val_conf_matrix.update(pred_labels.cpu().numpy(), seg_labels.cpu().numpy())

                lane_mask = (seg_labels > 0)
                if lane_mask.any():
                    val_lane_mae_total += torch.abs(depth_pred - depth_target)[lane_mask].sum().item()
                val_depth_mae_total += torch.abs(depth_pred - depth_target).sum().item()
                val_count += img.size(0) * img.shape[2] * img.shape[3]

        val_epoch_loss = val_loss / len(val_data.dataset)
        val_miou = val_conf_matrix.iou().mean()
        val_depth_mae = val_depth_mae_total / val_count
        val_lane_mae = val_lane_mae_total / val_count

        logger.add_scalar('val_loss', val_epoch_loss, epoch)
        logger.add_scalar('val_mIoU', val_miou, epoch)
        logger.add_scalar('val_depth_MAE', val_depth_mae, epoch)
        logger.add_scalar('val_lane_MAE', val_lane_mae, epoch)

        print(f"Epoch {epoch+1:2d}/{num_epoch:2d} | "
              f"Train Loss: {epoch_loss:.4f} | mIoU: {train_miou:.3f} | Depth MAE: {train_depth_mae:.4f} | Lane MAE: {train_lane_mae:.4f} || "
              f"Val Loss: {val_epoch_loss:.4f} | mIoU: {val_miou:.3f} | Depth MAE: {val_depth_mae:.4f} | Lane MAE: {val_lane_mae:.4f}")

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