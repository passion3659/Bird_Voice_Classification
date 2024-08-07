## Bird_Voice_Classification
This repository is the 2st place solution for DACON contest Bird_Voice_Classification monthly AI Contest

## Overview
현재 진행중인 연구가 음성관련이라 좋은 경험을 쌓고자 대회에 임하게 되었습니다. 
바빠서 다른 여러 다양한 시도를 못한게 아쉬웠지만 혼자서 scratch로 딥러닝 파이프라인을 구축해 의미 있는 대회였던거 같습니다.
pytorch-lighting의 문법을 주로 사용하였습니다. 

- 기법들을 순서대로 정리하자면 아래와 같습니다.
    - 동일한 length를 유지하기 위한 데이터 전처리
    - mel spectrogram 사용 (n_fft=1024, n_mels=128)
    - ResNetish 모델 사용


## Train & Test
```python
python train.py --epochs 20 --batch_size 32 --exp_dir exp10

python test.py --output_filename submission10.csv
```
