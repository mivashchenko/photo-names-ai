import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image: np.ndarray):
    raw_image = Image.fromarray(image).convert('RGB')

    text = "the image of"

    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iFace = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title='Image Captioning',
    description='This is a simple web for generating captions for images.'
)

iFace.launch()
