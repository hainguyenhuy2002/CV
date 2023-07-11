from super_image import ImageLoader
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
import requests

def detect_from_url(image_url, model):
    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = ImageLoader.load_image(image)

    model.eval()

    with torch.no_grad:
        preds = model(inputs)

    ImageLoader.save_image(preds, './scaled_2x.png')
    ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')

def detect_from_path(image_path, model):
    # Load the image from the given image_path
    image = Image.open(image_path)

    # Load the image into a format that the model can understand
    inputs = ToTensor()(image).unsqueeze(0)

    # Use the given model to make predictions
    model.eval()

    with torch.no_grad:
        preds = model(inputs)

    # Convert the predictions into an image
    output_image = ToPILImage()(preds.squeeze(0))

    # Save the scaled image and a comparison image
    output_image.save('./scaled_2x.png')
    ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')

# Test
# url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
# detect_from_url(url, model)