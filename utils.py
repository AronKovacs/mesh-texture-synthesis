from jinja2 import Template
import numpy as np
from PIL import Image
from pycuda.compiler import SourceModule

def pycuda_init():
    import pycuda.autoinit

def compile_cuda_module(source, no_extern_c=False, **kwargs):
    template = Template(source)
    rendered_template = template.render(kwargs)
    return SourceModule(rendered_template, no_extern_c=no_extern_c)

def save_image(data, path):
    if len(data.shape) != 3:
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = np.reshape(data, data.shape[1:])
        else:
            raise Exception('Invalid data dimension')
    if data.shape[-1] == 3:
        channels = 'RGB'
    elif data.shape[-1] == 4:
        channels = 'RGBA'
    else:
        raise Exception('Invalid number of channels, must be either 3 or 4')
    image = Image.fromarray((data * 255.0).astype('uint8'), channels)
    image.save(path)

def load_image(path, dtype, force_format=None, add_batch_dim=False):
    if force_format is not None and force_format != 'RGB' and force_format != 'RGBA':
        raise Exception('force_format muse be either None, "RGB" or "RGBA"')
    if dtype != np.uint8 and dtype != np.float32 and dtype != np.float64:
        raise Exception('dtype must be either np.uint8, np.float32, np.float64')
    image = np.asarray(Image.open(path))

    if dtype != np.uint8:
        image = image.astype(dtype) / 255.0

    if image.shape[2] == 3 and force_format == 'RGBA':
        new_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)
        new_image[:, :, :3] = image
        new_image[:, :, 3] = 1.0
        image = new_image
    elif image.shape[2] == 4 and force_format == 'RGB':
        new_image = np.zeros((image.shape[0], image.shape[1], 3))
        new_image[:, :, :] = image[:, :, :3]
        image = new_image

    if add_batch_dim:
        image = np.reshape(image, (1, *image.shape))

    return image

def save(data, path):
    with open(path, 'wb') as f:
        np.save(f, data, allow_pickle=True)

def load(path):
    with open(path, 'rb') as f:
        return np.load(f, allow_pickle=True)