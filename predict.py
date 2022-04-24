import torch
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
from cog import BasePredictor, Path, Input, BaseModel

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


class Predictor(BasePredictor):
    def setup(self):
        opt_path_denoise = "options/test/SIDD/NAFNet-width64.yml"
        opt_denoise = parse(opt_path_denoise, is_train=False)
        opt_denoise["dist"] = False

        opt_path_deblur = "options/test/GoPro/NAFNet-width64.yml"
        opt_deblur = parse(opt_path_deblur, is_train=False)
        opt_deblur["dist"] = False

        opt_path_stereo = "options/test/NAFSSR/NAFSSR-L_4x.yml"
        opt_stereo = parse(opt_path_stereo, is_train=False)
        opt_stereo["dist"] = False

        self.models = {
            "Image Denoising": create_model(opt_denoise),
            "Image Debluring": create_model(opt_deblur),
            "Stereo Image Super-Resolution": create_model(opt_stereo),
        }

    def predict(
        self,
        task_type: str = Input(
            choices=[
                "Image Denoising",
                "Image Debluring",
                "Stereo Image Super-Resolution",
            ],
            default="Image Debluring",
            description="Choose task type.",
        ),
        image: Path = Input(
            description="Input image. Stereo Image Super-Resolution, upload the left image here.",
        ),
        image_r: Path = Input(
            default=None,
            description="Right Input image for Stereo Image Super-Resolution. Optional, only valid for Stereo"
            " Image Super-Resolution task.",
        ),
    ) -> Path:

        out_path = Path(tempfile.mkdtemp()) / "output.png"

        model = self.models[task_type]
        if task_type == "Stereo Image Super-Resolution":
            assert image_r is not None, (
                "Please provide both left and right input image for "
                "Stereo Image Super-Resolution task."
            )

            img_l = imread(str(image))
            inp_l = img2tensor(img_l)
            img_r = imread(str(image_r))
            inp_r = img2tensor(img_r)
            stereo_image_inference(model, inp_l, inp_r, str(out_path))

        else:

            img_input = imread(str(image))
            inp = img2tensor(img_input)
            out_path = Path(tempfile.mkdtemp()) / "output.png"
            single_image_inference(model, inp, str(out_path))

        return out_path


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def single_image_inference(model, img, save_path):
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    imwrite(sr_img, save_path)


def stereo_image_inference(model, img_l, img_r, out_path):
    img = torch.cat([img_l, img_r], dim=0)
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    img_L = visuals["result"][:, :3]
    img_R = visuals["result"][:, 3:]
    img_L, img_R = tensor2img([img_L, img_R], rgb2bgr=False)

    # save_stereo_image
    h, w = img_L.shape[:2]
    fig = plt.figure(figsize=(w // 40, h // 40))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.title("NAFSSR output (Left)", fontsize=14)
    ax1.axis("off")
    ax1.imshow(img_L)

    ax2 = fig.add_subplot(2, 1, 2)
    plt.title("NAFSSR output (Right)", fontsize=14)
    ax2.axis("off")
    ax2.imshow(img_R)

    plt.subplots_adjust(hspace=0.08)
    plt.savefig(str(out_path), bbox_inches="tight", dpi=600)
