import os
import torch
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from dataset import *
from models import *

import torch.utils.tensorboard as tb

class SupervisedTrainer:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainset     = MAIC(['adult'], args.fold, is_train=True)
        self.testset      = MAIC(['adult'], args.fold, is_train=False)
        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True , drop_last=True )
        self.test_loader  = DataLoader(self.testset , batch_size=args.batch_size, shuffle=False, drop_last=False)

        self.model        = CNN()
        self.model.to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion    = nn.MSELoss()
        
        self.train_loss = None
        self.test_loss  = None
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}') \
            if self.args.use_tb else None
        
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f'model name:{args.model}\ndataset:{args.dataset}\ndevice:{self.device}\nTensorboard:{self.args.use_tb}\nTotal parameter:{total_params:,}')
        

    def train(self):
        self.model.train()
        losses = []
        for data in self.train_loader:
            self.optimizer.zero_grad()
            X = [data['data'].to(self.device), data['gender'].to(self.device)]
            Y = data['age'].to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred.reshape(-1), Y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.train_loss = np.mean(losses)
        
        if self.args.use_tb:
            self.TB_WRITER.add_scalar("Train Loss", self.train_loss, self.epoch+1)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds, targets, losses = [], [], []
        for data in self.test_loader:
            X = [data['data'].to(self.device), data['gender'].to(self.device)]
            Y = data['age'].to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred.reshape(-1), Y)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        self.test_loss = np.mean(np.abs(targets-preds))
        # if self.args.use_tb:
        #     self.TB_WRITER.add_scalar(f'Test Loss', self.test_loss, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Test Accuracy', self.acc, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Sensitivity', sens, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'F1-Score', f1, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Specificity', spec, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Test Accuracy (Balanced)', bal_acc, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'AUROC', auroc, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss:>6.4f} Test Loss:{self.test_loss:>6.4f}', flush=True)

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = SupervisedTrainer(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0:
            trainer.save_model()
        trainer.epoch += 1