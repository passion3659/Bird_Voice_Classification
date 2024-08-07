import os
import pandas as pd
import argparse

import torch
from torch.utils.data import  DataLoader
import torchaudio

import pytorch_lightning as pl

from data_utils.dataset import AudioDataset
from model.model import ResNetish, pl_Module,BasicBlock, Net, CNN2

def main(args):
    torch.set_float32_matmul_precision('medium')
    
    # data 경로 확보
    full_path = os.getcwd()
    data_path = os.path.join(full_path, 'data')

    # csv 파일 경로 확보
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    sample_path = os.path.join(data_path, 'sample_submission.csv')

    # data loader 선언
    test_annotations = pd.read_csv(test_path)
    test_dataset = AudioDataset(test_annotations, root_dir=data_path, has_label=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    # model 선언
    model = ResNetish( BasicBlock, [2, 2, 2, 2], num_classes=1)

    # PyTorch Lightning 적용
    classifier = pl_Module(model=model, lr=0.001)
    
    # 저장된 모델 불러오기
    checkpoint = torch.load("/home/workspace/dacon/01_bird_voice_classification/exp/exp10/checkpoint/best-model-epoch=10-val_loss=0.32.ckpt")
    classifier.load_state_dict(checkpoint["state_dict"])
    
    # trainer선언후 test값 가져오기
    trainer = pl.Trainer()
    trainer.test(classifier, test_loader)
    test_results = trainer.model.get_test_results()
    test_numpy = test_results.cpu().numpy()
    
    # submission파일 만들기
    sample_submission = pd.read_csv(sample_path)
    sample_submission['label'] = test_numpy
    
    submission_path = os.path.join(full_path, 'submission_output')
    submission_path = os.path.join(submission_path, args.output_filename)
    sample_submission.to_csv(submission_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test an audio classification model and generate submission file.")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for testing')
    parser.add_argument('--output_filename', type=str, help='output filename for submission')

    args = parser.parse_args()
    pl.seed_everything(990627)
    main(args)