# Dataset

LSVSC_100.json
```
[
    {
        "wav": "00000.wav",
        "class": "[2F0NM]",
        "text": "Thưa các bạn, để có thể nhanh chóng biết tin về toàn bộ tình hình xảy ra, Bô khâu ở trong phòng nhân viên đợi cờ rê mơ trở về",
        "duration": 8.482563,
        "editedby": "B21DCVT013"
    },
    {
        "wav": "00001.wav",
        "class": "[2F0NM]",
        "text": "cờ rê mơ thông báo mọi điều cho Bô khâu biết",
        "duration": 2.55825,
        "editedby": "B21DCVT013"
    }
]
```
LSVSC_train.json
```
{
    "ex0": {
        "wav": "36688.wav",
        "class": "[0F0NY]",
        "text": "Đến lúc nhận được thì chỉ có nhận được có ba triệu rưỡi",
        "duration": 2.5405,
        "editedby": "B21DCCN497"
    },
    "ex1": {
        "wav": "25937.wav",
        "class": "[0M0NY]",
        "text": "Thay mặt ủy ban kiểm tra trung ương ông Trần Cẩm Tú chủ nhiệm ủy ban kiểm tra trung ương tiếp thu ý kiến chỉ đạo của đồng chí trưởng đoàn kiểm tra",
        "duration": 6.335687,
        "editedby": "B21DCCN599"
    }
}
```
Download the dataset from [here](https://drive.google.com/drive/folders/1tiPKaIOC7bt6isv5qFqf61O_2jFK8ZOI)

# Requirements
```
HyperPyYAML==1.1.0
numpy==1.23.5
speechbrain==0.5.15
torch==1.13.1
```
# Training

## Transformer
```
cd transformer_asr
python transformer_words_train.py hparams/Transformer.yaml --dataset_dir <path to dataset>
```
## Transformer with  SpecAugmentation
```
cd transformer_asr
python transformer_words_train.py hparams/Transformer_SpecAugment.yaml --dataset_dir <path to dataset>
```
## Transformer with Adaptive SpecAugmentation
```
cd transformer_asr
python transformer_words_train.py hparams/Transformer_AdaptiveSpecAugment.yaml --dataset_dir <path to dataset>
```
## LAS
```
cd las_asr
python las_train.py hparams/LAS.yaml --dataset_dir <path to dataset>
```
## LAS with SpecAugmentation
```
cd las_asr
python las_train.py hparams/LAS_SpecAugment.yaml --dataset_dir <path to dataset>
```
