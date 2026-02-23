import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import utils.config as config
from utils.dataset import QaTa
from engine.wrapper import LanGuideMedSegWrapper
from tqdm import tqdm
from monai.metrics import DiceMetric
torch.set_float32_matmul_precision("high")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    return parser.parse_args()


def compute_metrics(pred, gt):
    """
    pred, gt: binary tensors (0/1)
    """

    pred = pred.view(-1)
    gt = gt.view(-1)

    TP = torch.sum((pred == 1) & (gt == 1)).item()
    TN = torch.sum((pred == 0) & (gt == 0)).item()
    FP = torch.sum((pred == 1) & (gt == 0)).item()
    FN = torch.sum((pred == 0) & (gt == 1)).item()

    eps = 1e-7

    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)

    return sensitivity, specificity, accuracy, iou, dice


if __name__ == "__main__":

    args_cli = get_parser()
    args = config.load_cfg_from_cfg_file(args_cli.config)

    print("CUDA:", torch.cuda.is_available())

    # 🔹 Test Dataset
    ds_test = QaTa(
        csv_path=args.valid_csv_path,   # change if you have test_csv
        root_path=args.valid_root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode='valid'
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 🔹 Load model
    model = LanGuideMedSegWrapper.load_from_checkpoint(
        args_cli.ckpt,
        args=args
    )

    model.cuda()
    model.eval()

    model = model.to("cuda")

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch"
    )
    torch.backends.cudnn.benchmark = True
    print("Model loaded")
    total_sens = 0
    total_spec = 0
    total_acc = 0
    total_iou = 0
    total_dice = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dl_test):

            (image, text), gt = batch

            image = image.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            text = {
                "input_ids": text["input_ids"].cuda(non_blocking=True),
                "attention_mask": text["attention_mask"].cuda(non_blocking=True)
            }

            outputs = model([image, text])   # correct multimodal call
            preds = (outputs > 0.5).long()   # model already has sigmoid

            sens, spec, acc, iou, dice = compute_metrics(preds, gt)
            dice_metric(preds.float(), gt.float()) 
            total_sens += sens
            total_spec += spec
            total_acc += acc
            total_iou += iou
            total_dice += dice
            total_samples += 1

    final_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print("\n================ FINAL TEST METRICS ================")
    print(f"Dice        : {final_dice:.4f}")
    print("====================================================")
    print("\n================ FINAL TEST METRICS ================")
    print(f"Sensitivity : {total_sens / total_samples:.4f}")
    print(f"Specificity : {total_spec / total_samples:.4f}")
    print(f"Accuracy    : {total_acc / total_samples:.4f}")
    print(f"IoU         : {total_iou / total_samples:.4f}")
    print(f"Dice        : {total_dice / total_samples:.4f}")
    print("====================================================")