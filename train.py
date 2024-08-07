import os
import pandas as pd
import argparse

import torch
from torch.utils.data import  DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from data_utils.dataset import AudioDataset
from model.model import ResNetish, pl_Module, BasicBlock, Net, CNN2

# data경로 확보
def main(args):
    torch.set_float32_matmul_precision('medium')
    full_path = os.getcwd()
    data_path = os.path.join(full_path, 'data')

    # csv파일 경로 확보
    train_path = os.path.join(data_path, 'train.csv')

    # data loader 선언
    annotations = pd.read_csv(train_path)
    train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations['label'])

    train_dataset = AudioDataset(train_annotations, root_dir=data_path)
    val_dataset = AudioDataset(val_annotations, root_dir=data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, persistent_workers=True)

    # model 선언
    model = ResNetish(BasicBlock, [2, 2, 2, 2], num_classes=1)

    # PyTorch Lightning 적용
    classifier = pl_Module(model=model, lr=0.001)

    # callback 선언 (가장 성능 좋은 모델 저장)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath=os.path.join('./exp', args.exp_dir, 'checkpoint'),  
        filename='best-model-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=1, 
        mode='min',
        )
    
    # callback 선언 (가장 마지막 모델 저장)
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=os.path.join('./exp', args.exp_dir, 'checkpoint'),  
        filename='last-model-{step:06d}', 
        save_top_k=1,
        monitor=None 
        )

    callbacks = [checkpoint_callback, checkpoint_callback_last]

    # PyTorch Lightning Trainer 선언
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices="auto",
        callbacks=callbacks,
        limit_train_batches=1.0,
        )

    # 모델 훈련
    trainer.fit(classifier, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--exp_dir', type=str, help='experiment directory for saving checkpoints')

    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    pl.seed_everything(990627)

    main(args)