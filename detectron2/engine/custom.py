
import torch
import pickle
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from google.colab.patches import cv2_imshow
__all__ = ["CustomPredictor"]
class CustomPredictor:
    def __init__(self, model):
        if type(model) == str:
            self.model = torch.load(model)
            
        else:
            self.model = model

        self.model.eval()
    def load_metadata(self, meta):
        if type(meta) == str:
            self.metadata = pickle.load(open(meta, 'rb'))
        else:
            self.metadata = meta

    def predict(self, image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            #if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
            original_image = image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            new_image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": new_image, "height": height, "width": width}
            outputs = self.model([inputs])[0]
            return outputs

    def predict_and_visualize(self, image, display=False):
        outputs  = self.predict(image)
        v = Visualizer(image[:, :, ::-1],
                scale=0.8, 
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis = v.get_image()[:, :, ::-1]
        if not display:
            return vis
        cv2_imshow(vis)
