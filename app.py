import os
import time
import numpy as np
import gradio as gr
from glob import glob
from functools import partial
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from transformers import SegformerForSemanticSegmentation

@dataclass
class Configs:
    NUM_CLASSES: int = 4  # including background.
    CLASSES: tuple = ("Large bowel", "Small bowel", "Stomach")
    IMAGE_SIZE: tuple[int, int] = (288, 288)  # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)
    MODEL_PATH: str = os.path.join(os.getcwd(), "segformer_trained_weights")

def get_model(*, model_path, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(model_path, num_labels=num_classes, ignore_mismatched_sizes=True)
    return model

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"

@torch.inference_mode()
def predict(input_image, model=None, preprocess_fn=None, device="cpu"):
    shape_H_W = input_image.size[::-1]
    input_tensor = preprocess_fn(input_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Measure inference time
    start_time = time.time()
    outputs = model(pixel_values=input_tensor.to(device), return_dict=True)
    end_time = time.time()
    inference_time = end_time - start_time
    
    predictions = F.interpolate(outputs["logits"], size=shape_H_W, mode="bilinear", align_corners=False)
    preds_argmax = predictions.argmax(dim=1).cpu().squeeze().numpy()
    seg_info = [(preds_argmax == idx, class_name) for idx, class_name in enumerate(Configs.CLASSES, 1)]
    
    return (input_image, seg_info, inference_time)

if __name__ == "__main__":
    class2hexcolor = {"Stomach": "#007fff", "Small bowel": "#009A17", "Large bowel": "#FF0000"}
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(model_path=Configs.MODEL_PATH, num_classes=Configs.NUM_CLASSES)
    model.to(DEVICE)
    model.eval()
    _ = model(torch.randn(1, 3, *Configs.IMAGE_SIZE[::-1], device=DEVICE))
    
    model_size = get_model_size(model)
    
    preprocess = TF.Compose(
        [
            TF.Resize(size=Configs.IMAGE_SIZE[::-1]),
            TF.ToTensor(),
            TF.Normalize(Configs.MEAN, Configs.STD, inplace=True),
        ]
    )
    
    with gr.Blocks(title="Medical Image Segmentation") as demo:
        gr.Markdown("""<h1><center>Medical Image Segmentation with UW-Madison GI Tract Dataset</center></h1>""")
        gr.Markdown(f"Model size: {model_size}")
        with gr.Row():
            img_input = gr.Image(type="pil", height=360, width=360, label="Input image")
            img_output = gr.AnnotatedImage(label="Predictions", height=360, width=360, color_map=class2hexcolor)
        inference_time = gr.Textbox(label="Inference Time (seconds)")
        section_btn = gr.Button("Generate Predictions")
        
        def process_and_time(input_image):
            image, seg_info, inf_time = predict(input_image, model=model, preprocess_fn=preprocess, device=DEVICE)
            return (image, seg_info), f"{inf_time:.4f}"
        
        section_btn.click(process_and_time, img_input, [img_output, inference_time])
        
        images_dir = glob(os.path.join(os.getcwd(), "samples") + os.sep + "*.png")
        examples = [i for i in np.random.choice(images_dir, size=10, replace=False)]
        gr.Examples(examples=examples, inputs=img_input, outputs=[img_output, inference_time])
    
    demo.launch()