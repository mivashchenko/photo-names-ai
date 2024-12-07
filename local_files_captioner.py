import os
import glob
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_dir = './images'

image_exts = ['jpg', 'jpeg', 'png']

with open("captions.txt", "w") as caption_file:
    for img_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f'*.{img_ext}')):
            raw_image = Image.open(img_path).convert('RGB')

            inputs = processor(raw_image, return_tensors="pt")

            out = model.generate(**inputs, max_new_tokens=50)

            caption  = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")