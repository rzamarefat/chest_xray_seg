import archs
import torch
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
from ultralytics import YOLO
import os
import gdown

class LungSeg:
    def __init__(self, device="cuda", model_name="unet++"):
        self._device = device
        self._model_name = model_name

        os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)
        if self._model_name == "unet++":
            self._build_unetpp()
        elif self._model_name == "yolov11":
            self._build_yolo()
        else:
            raise RuntimeError("Please provide a valid name for model_name")

    def _build_unetpp(self):
        ckpt_path = os.path.join(os.getcwd(), "weights", "unetpp_large_lung_seg.pth")

        if not(os.path.isfile(ckpt_path)):
            ckpt_url="https://drive.google.com/uc?id=15nnzaU3MOlBTg49-QqMb6VnmvwiOuaDc"
            gdown.download(ckpt_url, ckpt_path, quiet=False)

        self._model = archs.__dict__["NestedUNet"](1,3,False)
        self._model.to(self._device)
        self._model.load_state_dict(torch.load(ckpt_path))
        self._model.eval()

        self._val_transform = Compose([
                Resize(256, 256),
                Normalize(),
            ])
        
    def _build_yolo(self):
        ckpt_path = os.path.join(os.getcwd(), "weights", "yolov11_large_lung_seg.pt")
        if not(os.path.isfile(ckpt_path)):
            ckpt_url="https://drive.google.com/uc?id=1O7sRzdD47arMtVWRs-33dM0dQLVF1_wS"
            gdown.download(ckpt_url, ckpt_path, quiet=False)

        self._model = YOLO(ckpt_path)

    def _preprocess(self, images):
        tensors = []

        for img in images:
            transformed = self._val_transform(image=img)
            transformed = transformed["image"].astype('float32') / 255
            transformed = transformed.transpose(2, 0, 1)
            transformed_tensor = torch.Tensor(transformed)
            
            tensors.append(transformed_tensor)
        batch_tensor = torch.stack(tensors)

        return batch_tensor
    
    def _segment_using_unetpp(self, imgs):
        print("Using UNET++ to segment...")
        
        model_input = self._preprocess(imgs)

        with torch.no_grad():
            model_input = model_input.to(self._device)
            output = self._model(model_input)
            output = torch.sigmoid(output).cpu().numpy()

            
            predicted_masks_holder = []
            for i in range(len(output)):
                for c in range(1):
                    predicted_masks_holder.append((output[i, c] * 255).astype('uint8'))

        return predicted_masks_holder
    
    def _segment_using_yolo(self, imgs):
        print("Using YOLOv1 to segment...")
        preds = self._model.predict(imgs, retina_masks=True)
        predicted_masks_holder = []
        for pred in preds:
            mask = pred.masks.data.cpu().numpy()[0]
            mask *= 255
            mask = mask.astype('uint8')

            predicted_masks_holder.append(mask)

        return predicted_masks_holder



    def __call__(self, imgs):
        print(type(imgs))
        if self._model_name == "unet++":
            predicted_masks_holder = self._segment_using_unetpp(imgs)
        elif self._model_name == "yolov11":
            predicted_masks_holder = self._segment_using_yolo(imgs)

        print(predicted_masks_holder[0].shape)
        return predicted_masks_holder