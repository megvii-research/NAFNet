"""Metrics on exactly the RGB uint8 pixels saved in enhanced PNG files."""
import warnings
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
from basicsr.metrics.niqe import niqe
from basicsr.metrics.metric_util import to_y_channel

PARAMS = Path(__file__).resolve().parents[1]/'basicsr/metrics/niqe_pris_params.npz'


def score_image(prediction, reference=None, crop_border=0):
    if crop_border < 0 or min(prediction.shape[:2]) <= 2*crop_border:
        raise ValueError('invalid crop_border')
    image = prediction[crop_border:-crop_border, crop_border:-crop_border] if crop_border else prediction
    result = {}
    if reference is not None:
        if reference.shape != prediction.shape:
            raise ValueError('metric dimensions differ')
        target = reference[crop_border:-crop_border, crop_border:-crop_border] if crop_border else reference
        mse = np.mean((image.astype(np.float64)-target.astype(np.float64))**2)
        result.update(psnr=float(10*np.log10(255**2/mse)) if mse else float('inf'), psnr_status='ok')
        if min(image.shape[:2]) < 11:
            result.update(ssim=None, ssim_status='unavailable: spatial size below 11x11')
        else:
            value = float(structural_similarity(image, target, data_range=255, channel_axis=2,
                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11))
            if not np.isfinite(value):
                raise ValueError('nonfinite SSIM')
            result.update(ssim=value, ssim_status='ok')
    blocks = (image.shape[0]//96)*(image.shape[1]//96)
    if blocks < 2:
        result.update(niqe=None, niqe_status='unavailable: fewer than two 96x96 blocks; no resizing used')
    else:
        try:
            with np.load(PARAMS) as params, warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                y = to_y_channel(image[..., ::-1].copy()).squeeze().astype(np.float32)
                value = np.asarray(niqe(y, params['mu_pris_param'], params['cov_pris_param'],
                                        params['gaussian_window'])).item()
            if not np.isfinite(value):
                raise ValueError('nonfinite NIQE')
            result.update(niqe=value, niqe_status='ok')
        except (ValueError, FloatingPointError, RuntimeWarning, np.linalg.LinAlgError) as error:
            result.update(niqe=None, niqe_status=f'unavailable: {type(error).__name__}: {error}')
    return result


def summarize(rows, paired):
    result = {'images': len(rows), 'metrics': {}}
    for metric in (('psnr', 'ssim', 'niqe') if paired else ('niqe',)):
        values = [r[metric] for r in rows if r[metric] is not None]
        result['metrics'][metric] = {'mean': float(np.mean(values)) if values else None,
                                     'valid': len(values), 'unavailable': len(rows)-len(values)}
    return result
