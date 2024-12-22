import archs
import torch
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
import cv2
import numpy as np

class LungSeg:
    def __init__(self, device="cuda"):
        self._device = device
        self._model = archs.__dict__["NestedUNet"](1,3,False)
        self._model.to(self._device)
        self._model.load_state_dict(torch.load(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\pytorch-nested-unet\models\dsb2018_96_NestedUNet_woDS\model.pth"))
        self._model.eval()

        self._val_transform = Compose([
                Resize(256, 256),
                Normalize(),
            ])

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

    def __call__(self, imgs):

        # cv2_imgs = []
        # for img in imgs:
        #     bgr_image = np.array(img)[..., ::-1]
        #     # bgr_image = Image.fromarray(bgr_image)
        #     cv2_imgs.append(bgr_image)
        
        model_input = self._preprocess(imgs)

        print(model_input.shape)

        with torch.no_grad():
            model_input = model_input.to(self._device)
            output = self._model(model_input)
            output = torch.sigmoid(output).cpu().numpy()

            
            predicted_masks_holder = []
            for i in range(len(output)):
                for c in range(1):
                    predicted_masks_holder.append((output[i, c] * 255).astype('uint8'))

        return predicted_masks_holder
