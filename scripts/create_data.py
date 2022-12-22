import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16,
    ).to("cuda")

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def get_depth(image):
    input_batch = transform(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def get_image(depth):
    return pipe(prompt='', image=depth, strength=0.5).images[0]


def get_true_image(fp):
    return Image.open(fp)


if __name__ == "__main__":
    im_fps = ["data/1.png", "data/2.png", "data/3.png"]
    for fp in im_fps:
        im = get_true_image(fp)
        depth = get_depth(im)
        im = get_image(depth)
        im = Image.fromarray(np.uint8(im * 255))
        im.save(fp.replace(".png", "_out.png"))