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
            # transformed_tensor = transformed_tensor.permute(2, 0, 1)


            print("transformed_tensor.shape", transformed_tensor.shape)

            
            tensors.append(transformed_tensor)

        # Stack all tensors into a single batch tensor
        batch_tensor = torch.stack(tensors)
        return batch_tensor


        #     # Normalize pixel values to [0, 255] for visualization (if needed)
        #     processed_image = np.clip(transformed_image * 255, 0, 255).astype(np.uint8)

        #     # Convert back to PIL.Image for display
        #     processed_images.append(Image.fromarray(processed_image))

        # return processed_images

    def __call__(self, pil_imgs):
        
        model_input = self._preprocess(pil_imgs)

        print(model_input.shape)

        with torch.no_grad():
            model_input = model_input.to(self._device)
            output = self._model(model_input)
            print("output.shape11", output.shape)
            output = torch.sigmoid(output).cpu().numpy()

            

            for i in range(len(output)):
                for c in range(1):
                    print("output[i, c].shape", output[i, c].shape)
                    print("np.max(output[i, c] * 255)", np.max(output[i, c] * 255))
                    cv2.imwrite(f"mask_{i}.png",
                                (output[i, c] * 255).astype('uint8'))


        # with torch.no_grad():
        #     for input, target, meta in val_loader:
        #         input = input.cuda()
        #         target = target.cuda()
        #         output = self._model(input)

                

                # output = torch.sigmoid(output).cpu().numpy()
                # for i in range(len(output)):
                #     cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                #                 (output[i, 0] * 255).astype('uint8'))

                # for i in range(len(output)):
                #     cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),(output[i, c] * 255).astype('uint8'))

if __name__ == "__main__":
    # from PIL import Image

    pil_img_1 = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\Chest-X-Ray\Chest-X-Ray\image\1000.png")
    pil_img_2 = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\Chest-X-Ray\Chest-X-Ray\image\1001.png")
    pil_img_3 = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\chest_xray_seg\Chest-X-Ray\Chest-X-Ray\image\1002.png")
    LungSeg()([pil_img_1, pil_img_2, pil_img_3])