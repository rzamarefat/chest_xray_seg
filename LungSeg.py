import archs
import torch
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
import cv2
import numpy as np
from ultralytics import YOLO

class LungSeg:
    def __init__(self, device="cuda", model_name="unetpp"):
        self._device = device
        self._model_name = model_name

        if self._model_name == "unetpp":
            self._build_unetpp()
        elif self._model_name == "yolo":
            self._build_yolo()
        else:
            raise RuntimeError("Please provide a valid name for model_name")

    def _build_unetpp(self):
        self._model = archs.__dict__["NestedUNet"](1,3,False)
        self._model.to(self._device)
        self._model.load_state_dict(torch.load(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\pytorch-nested-unet\models\dsb2018_96_NestedUNet_woDS\model.pth"))
        self._model.eval()

        self._val_transform = Compose([
                Resize(256, 256),
                Normalize(),
            ])
        
    def _build_yolo(self):
        self._model = YOLO(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\yolo\runs\segment\large\weights\best.pt")

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
        # cv2_imgs = []
        # for img in imgs:
        #     bgr_image = np.array(img)[..., ::-1]
        #     # bgr_image = Image.fromarray(bgr_image)
        #     cv2_imgs.append(bgr_image)
        
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
        if self._model_name == "unetpp":
            predicted_masks_holder = self._segment_using_unetpp(imgs)
        elif self._model_name == "yolo":
            predicted_masks_holder = self._segment_using_yolo(imgs)

        print(predicted_masks_holder[0].shape)
        return predicted_masks_holder
            

            

        
if __name__ == "__main__":
    seg = LungSeg(model_name="yolo")
    img_1 = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\YOLO_format\val\1000.png")
    img_2 = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\YOLO_format\val\1020.png")
    predicted_masks_holder = seg([img_1, img_2])
    print(predicted_masks_holder)