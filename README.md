# Proper Face Mask Object Detection
##### a [PyTorch](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) implementation, using [MobileNetV3](https://arxiv.org/abs/1905.02244) and [SSDLite](https://arxiv.org/abs/1801.04381)

# Model Results
<p align="center">
  <img alt="Result gif" align="center" src="example.gif" width="400"/>
</p>

# Using our trained model (CPU Inference)
the model was created with mobile deployment in mind, so it can be easily deployed on any machine with a webcam, following a few simple steps:

1. (optional, but recommended) create a clean Miniconda environment:
    ```
    conda create --name app python=3.8
    conda activate app
    ```
1. install requirements:
    ```
    pip install -r requirements_lite.txt
    ```
1. run the app:
    ```
    python app.py
    ```
1. (optional) the app can also be packed into a `.exe` file using `pyinstaller`:
    ```
    pip install pyinstaller
    pyinstaller --onefile -w app.py
    ```
    1. an antivirus exception may be needed for the `.exe` file.
    1. the `.exe` file should be ~270MB if using a clean Miniconda environment.

# Re-Training the model

## Enviorment
for training, use the `environment.yml` virtual environment:
```
conda env create -f environment.yml
```


## Data
Training is done using 2 Kaggle datasets:
1. [**face-mask-detection**](https://www.kaggle.com/andrewmvd/face-mask-detection) by **andrewmvd**
2. [**face-mask-detection-dataset**](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset) by **wobotintelligence**

data can be downloaded manually, or using the `kaggle` API:

1. install the `kaggle` package:
    ```
    pip install kaggle
    ```

2. create your Kaggle API key (in the Kaggle settings)
3. place your Kaggle API key `.json` file in `~/.kaggle/kaggle.json`
4. download and extract the data:

    ```
    kaggle datasets download wobotintelligence/face-mask-detection-dataset
    unzip face-mask-detection-dataset.zip -d data/wobotintelligence
    rm face-mask-detection-dataset.zip

    kaggle datasets download andrewmvd/face-mask-detection
    unzip face-mask-detection.zip -d data/andrewmvd
    rm face-mask-detection.zip
    ```

## Training (only single GPU supported)
training hyperparameters can be changed via the `config.py` file, for training run:
```
python main.py
```

or run `main.ipynb` if you preffer using a jupyter notebook.

## Model
1. we use the torchvision implementation of the **SSDLite-320x320** object detection model with a **MobileNetV3-Large** backbone: `torchvision.models.detection.ssdlite320_mobilenet_v3_large`.
1. we use a pretrained backbone, the object detection model is trained from scratch.
