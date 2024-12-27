# Lung Segmentation On Chest Xray Images

## Models

- [x] UNET++
- [x] YOLOv11

## Training Results
- UNET++
![unet++_train_val_loss](https://github.com/user-attachments/assets/8ce5aaef-eaab-4a23-af9a-ee2d44a98c43)
- YOLOv11
![yolov11_train_val_loss](https://github.com/user-attachments/assets/2280d48d-36c7-458e-9d25-ffb7b0577b19)

## Visual Results on Test Samples
![demo_masks](https://github.com/user-attachments/assets/f806cd80-89e6-4307-952d-2b49984345b8)

## Dataset
The dataset I used is [1](https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation)

## Use Streamlit Demo UI
1. Clone the repo, cd into the project and create a python env
```
python -m venv ./venv
```
2. Run the following command and wait for the dependencies to be installed
```
pip install -r requirements
```
3. Run the following
```
streamlit run app.py
```
