# Birds-classification

### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluate](#evaluate)
- [Inference](#inference)

## Installation

- Python 3.9.0
- All dependencies can be installed using following command:
```Shell
pip install -r requirements.txt
```

## Training 
Base model [model](https://drive.google.com/file/d/12vU1smGO2ib9TqTX8ER0sSIRP7ZLp0aJ/view?usp=share_link)
(You have to modify output layer!)(or download modified)
1. Prepare training data:

    -- download [dataset](https://drive.google.com/drive/folders/1u8NoMofPoogRtRUeAPZwN-ViazIMcjcV?usp=share_link)
    
    -- put the 'data' directory to the root project directory 

2. Train the model using dataset:

    Just run the train script: 
```Shell
python train.py
```
Also you can use optional parameters such as:

    --data "path to dataset" default='/data'
    --batch_size "batch size" default=32
    --num_epochs "number of epochs" default=50
    --save_model "path where model will be saved" default='/models'
    --save_logs "path where logs will be saved" default='/logs'

If you do not wish to train the model, you can download [pre-trained model](https://drive.google.com/file/d/1ANwmg-c41310YWmrM_vULX3-1wgJVJkB/view?usp=share_link) and save it in '/models'.

## Evaluate

To evaluate model just run the evaluate script:
```Shell
python evaluate.py
```

Also you can use optional parameters such as:

    --data "path to dataset" default='/data'
    --batch_size "batch size" default=32
    --load_model "path to load model" default='/models/best_model.h5'

## Inference

To inference model just run the inference script:

```Shell
python inference.py
```

Also you can use optional parameters such as:

    --image_path "path to image" default='/inference/0.jpg'
    --load_model "path to load model" default='/models/best_model.h5'
    --image_name "name of image in inference directory" default='0.jpg'

## References
- [BaseModel](https://www.kaggle.com/code/fareselmenshawii/birds-450-species-images-classification/data)
- [Report](https://docs.google.com/document/d/1YYghV4CnQHWNwT5YkpxcOrAeg120iaUkn1QZG19s77Q/edit?usp=sharing)