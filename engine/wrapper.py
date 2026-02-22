from utils.model import LanGuideMedSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime

class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(LanGuideMedSegWrapper, self).__init__()
        print("Model initiated")
        self.model = LanGuideMedSeg(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.patience = args.patience
        
        # self.save_hyperparameters()

    # def configure_optimizers(self):

    #     optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.hparams.lr)
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

    #     return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-6
        )

        return [optimizer], [scheduler]
    
    def forward(self,x):
       
       return self.model.forward(x)


    # def shared_step(self,batch,batch_idx):
    #     x, y = batch
    #     preds = self(x)
    #     loss = self.loss_fn(preds,y)
    #     return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}    
    
    def shared_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        return {
            'loss': loss,
            'preds': preds.detach(),   # integer tensor
            'y': y.long().detach()     # ensure integer target
        }
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    # def shared_step_end(self,outputs,stage):
    #     metrics = self.train_metrics if stage=="train" else (
    #         self.val_metrics if stage=="val" else self.test_metrics)
    #     for name in metrics:
    #         step_metric = metrics[name](outputs['preds'], outputs['y']).item()
    #         if stage=="train":
    #             self.log(name,step_metric,prog_bar=True)
    #     return outputs["loss"].mean()
    def shared_step_end(self, outputs, stage):

        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)

        loss = outputs["loss"]

        for name in metrics:
            metric_value = metrics[name](outputs['preds'], outputs['y'])

            if stage == "train":
                self.log(f"train_{name}",
                        metric_value,
                        prog_bar=True,
                        on_step=True,
                        on_epoch=False)

        if stage == "train":
            self.log("train_loss",
                    loss,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False)

        return loss
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    # def training_epoch_end(self, outputs):
    #     dic = self.shared_epoch_end(outputs,stage="train")
    #     self.print(dic)
    #     dic.pop("epoch",None)
    #     self.log_dict(dic, logger=True)
    
    def training_epoch_end(self, outputs):

        dic = self.shared_epoch_end(outputs, stage="train")
        self.print_bar()
        self.print(dic)

        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        if hasattr(self, "best_dice"):
            self.print(f"Best Val Dice So Far: {self.best_dice:.4f}")

    # def validation_epoch_end(self, outputs):
    #     dic = self.shared_epoch_end(outputs,stage="val")
    #     self.print_bar()
    #     self.print(dic)
    #     dic.pop("epoch",None)
    #     self.log_dict(dic, logger=True)
        
    #     #log when reach best score
    #     ckpt_cb = self.trainer.checkpoint_callback
    #     monitor = ckpt_cb.monitor 
    #     mode = ckpt_cb.mode 
    #     arr_scores = self.get_history()[monitor]
    #     best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
    #     if best_score_idx==len(arr_scores)-1:   
    #         self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
    #             monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def validation_epoch_end(self, outputs):

        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar()
        self.print(dic)

        # Remove epoch key for logging
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        # Get current val_dice
        current_dice = dic["val_dice"]

        # Initialize best dice tracking
        if not hasattr(self, "best_dice"):
            self.best_dice = current_dice
            self.no_improve_count = 0
        else:
            if current_dice > self.best_dice:
                self.best_dice = current_dice
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

        # Print tracking info
        self.print("\n" + "="*80)
        self.print(f"Epoch {self.current_epoch}")
        self.print(f"Val Loss  : {dic['val_loss']:.4f}")
        self.print(f"Val Dice  : {dic['val_dice']:.4f}")
        self.print(f"Val IoU   : {dic['val_MIoU']:.4f}")
        self.print(f"Val Acc   : {dic['val_acc']:.4f}")
        self.print(f"Best Dice : {self.best_dice:.4f}")
        self.print(f"EarlyStop : {self.no_improve_count}/{self.patience}")
        self.print("="*80)

        # Print best model path if improved
        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb.best_model_path:
            self.print(f"💾 Best model saved at: {ckpt_cb.best_model_path}")


    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)