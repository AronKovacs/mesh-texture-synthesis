import utils

from tmscnn import Tmscnn
from tmscnn_layers import *

import numpy as np
from PIL import Image

import os
from pathlib import Path
import sys

utils.pycuda_init()
tf.config.run_functions_eagerly(True)

np.random.seed(0)
tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

with tf.device('/device:GPU:0'):
    small_allocation = tf.zeros((1,))

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

save_progress = True
save_each_nth_step = 50

obj_path = sys.argv[1]
obj_name = Path(obj_path).stem

content_path = sys.argv[2]
content_name = Path(content_path).stem

style_path = sys.argv[3]
style_name = Path(style_path).stem
style_image = np.asarray(Image.open(style_path)) / 255.0
style_image = np.reshape(style_image, (1, *style_image.shape))
style_image = style_image.astype(np.float32)
style_image = style_image[:, :, :, :3]

style_width = style_image.shape[1]
style_height = style_image.shape[2]

content_image = np.asarray(Image.open(content_path)) / 255.0
content_image = np.reshape(content_image, (1, *content_image.shape))
content_image = content_image.astype(np.float32)
content_image = content_image[:, :, :, :3]

width = content_image.shape[1]
height = content_image.shape[2]
channels = 3

content_layers = ['block4_conv2'] 

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

tmscnn_gpu = TmscnnGpu()
tmscnn = Tmscnn()

curved_neighborhood_plane = tmscnn.create_plane_neighborhoods(0, (style_width, style_height), 6)

mesh_ids = [0]
sources_nearest = [curved_neighborhood_plane['sources_nearest']]
sources_linear = [curved_neighborhood_plane['sources_linear']]
scaling_factors = [curved_neighborhood_plane['scaling_factors']]
pooling_spans = [curved_neighborhood_plane['pooling_spans']]
pooling_indices = [curved_neighborhood_plane['pooling_indices']]
group_textures = [curved_neighborhood_plane['group_textures']]

tmscnn_gpu.use_computation_buffers(mesh_ids, sources_nearest, sources_linear, scaling_factors, pooling_spans, pooling_indices, force_rewrite=True)

class Model(tf.keras.models.Model):
    def __init__(self, tmscnn, tmscnn_gpu, width, height, channels, style_layers, content_layers):
        super(Model, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.inp = tf.keras.layers.InputLayer(input_shape = (width, height, channels))

        current_layer = 0

        self.tmscnn_layers = []

        self.textures_to_groups = TMSCNNTexturesToFlattenedGroups(tmscnn)

        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 64, (3,3), interpolation='linear', name='block1_conv1', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 64, (3,3), interpolation='nearest', name='block1_conv2', activation='relu'))
        
        self.tmscnn_layers.append(TMSCNNAveragePoolingGPU(tmscnn, tmscnn_gpu, current_layer, name='block1_pool'))
        current_layer += 1

        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 128, (3,3), interpolation='nearest', name='block2_conv1', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 128, (3,3), interpolation='nearest', name='block2_conv2', activation='relu'))

        self.tmscnn_layers.append(TMSCNNAveragePoolingGPU(tmscnn, tmscnn_gpu, current_layer, name='block2_pool'))
        current_layer += 1
        
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 256, (3,3), interpolation='nearest', name='block3_conv1', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 256, (3,3), interpolation='nearest', name='block3_conv2', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 256, (3,3), interpolation='nearest', name='block3_conv3', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 256, (3,3), interpolation='nearest', name='block3_conv4', activation='relu'))
        
        self.tmscnn_layers.append(TMSCNNAveragePoolingGPU(tmscnn, tmscnn_gpu, current_layer, name='block3_pool'))
        current_layer += 1

        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block4_conv1', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block4_conv2', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block4_conv3', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block4_conv4', activation='relu'))
        
        self.tmscnn_layers.append(TMSCNNAveragePoolingGPU(tmscnn, tmscnn_gpu, current_layer, name='block4_pool'))
        current_layer += 1

        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block5_conv1', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block5_conv2', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block5_conv3', activation='relu'))
        self.tmscnn_layers.append(TMSCNNConvolutionGPU(tmscnn, tmscnn_gpu, current_layer, 512, (3,3), interpolation='nearest', name='block5_conv4', activation='relu'))
        
        self.tmscnn_layers.append(TMSCNNAveragePoolingGPU(tmscnn, tmscnn_gpu, current_layer, name='block5_pool'))
        current_layer += 1

    def call(self, inputs):
        style_output = []
        content_output = []

        mesh_ids = inputs[0]
        textures = inputs[1]

        x = self.inp(textures)

        x = self.textures_to_groups([mesh_ids, x])
        for layer in self.tmscnn_layers:
            x = layer([mesh_ids, x])
            if layer.name in self.style_layers:
                style_output.append(x)
            if layer.name in self.content_layers:
                content_output.append(x)

        output = style_output + content_output

        return output

def gram_matrix(input_tensor):
    result = tf.expand_dims(tf.linalg.einsum('ic,id->cd', input_tensor, input_tensor), axis=0)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[0], tf.float32)
    result /= num_locations
    return result

vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

class ExtractorModel(tf.keras.models.Model):
    def __init__(self, model_args, style_layers, content_layers, width, height, channels):
        super(ExtractorModel, self).__init__()
        self.model = Model(*model_args, style_layers, content_layers)

        # initializing the weights
        batch_mesh_ids = np.array([[0]], np.int32)
        batch_input_textures = tf.zeros((1, width, height, channels), np.float32)
        inputs = [tf.convert_to_tensor(batch_mesh_ids), batch_input_textures]
        _ = self.model(inputs)

        for layer_vgg19 in vgg19.layers:
            if 'conv' not in layer_vgg19.name:
                continue
            for layer_model in self.model.layers:
                if layer_vgg19.name == layer_model.name:
                    w = layer_vgg19.get_weights()[0]
                    w = np.swapaxes(w, 2, 3)
                    w = np.reshape(w, (9, w.shape[2], w.shape[3]))
                    layer_model.w.assign(w)
                    layer_model.b.assign(layer_vgg19.get_weights()[1])

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.model.trainable = False

    def call(self, inputs):
        inputs[1] = tf.keras.applications.vgg19.preprocess_input(inputs[1] * 255.0)

        outputs = self.model(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

print('Extracting style...')
extractor_model = ExtractorModel((tmscnn, tmscnn_gpu, style_width, style_height, channels), style_layers, content_layers, style_width, style_height, channels)
batch_mesh_ids = np.array([[0]], np.int32)
style_targets = extractor_model([batch_mesh_ids, tf.constant(style_image)])['style']

def normalize_mesh(vertices):
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)

    bounds_diff = max_bounds - min_bounds
    scale = np.max(bounds_diff)
    return (vertices - min_bounds) / scale

def load_obj(path):
    obj_model = tmscnn.load_obj(path)
    object_world_positions = obj_model['world_positions']
    object_indices = obj_model['world_position_indices']

    object_uvs = obj_model['uvs']
    object_uv_indices = obj_model['uv_indices']

    object_world_positions = normalize_mesh(object_world_positions)

    object_normals = tmscnn.compute_normals(object_world_positions, object_indices)
    return object_world_positions, object_uvs, object_normals, object_indices, object_uv_indices

world_positions, uvs, normals, indices, uv_indices = load_obj(obj_path)

print('Precomputation...')

tangents = tmscnn.create_tangent_vectors(world_positions, indices, (1, 0, 0), (0, 1, 0))
curved_neighborhood = tmscnn.create_curved_neighborhoods(
    0,
    world_positions,
    tangents,
    uvs,
    indices,
    uv_indices,
    (width, height),
    6
)

mesh_ids = [0]
sources_nearest = [curved_neighborhood['sources_nearest']]
sources_linear = [curved_neighborhood['sources_linear']]
scaling_factors = [curved_neighborhood['scaling_factors']]
pooling_spans = [curved_neighborhood['pooling_spans']]
pooling_indices = [curved_neighborhood['pooling_indices']]
group_textures = [curved_neighborhood['group_textures']]

for x in range(width):
    for y in range(height):
        if group_textures[0][0][x, y] < 0:
            content_image[0, x, y] = 0

extractor_model = ExtractorModel((tmscnn, tmscnn_gpu, width, height, channels), style_layers, content_layers, width, height, channels)

tmscnn_gpu.use_computation_buffers(mesh_ids, sources_nearest, sources_linear, scaling_factors, pooling_spans, pooling_indices, force_rewrite=True)

content_targets = extractor_model([batch_mesh_ids, tf.constant(content_image)])['content']

style_weight = 1
content_weight = 1e3

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss

    return loss

current_learning_rate = 0.1
opt = tf.optimizers.Adam(learning_rate=current_learning_rate)

def train_step(image):
    tmscnn_gpu.use_computation_buffers(mesh_ids, sources_nearest, sources_linear, scaling_factors, pooling_spans, pooling_indices, force_rewrite=False)
    with tf.GradientTape() as tape:
        outputs = extractor_model([batch_mesh_ids, image])
        loss = style_content_loss(outputs)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

batch_mesh_ids = tf.Variable(np.array([[0]], np.int32))
image = tf.Variable(content_image)

print("Optimization:")

step = 0
n_steps = 500
halve_lr_after_nth = 200
for _ in range(n_steps):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
    if step % halve_lr_after_nth == 0:
        current_learning_rate /= 2
        opt.lr.assign(current_learning_rate)
    if save_progress and step % save_each_nth_step == 0:
        img = image.numpy()
        utils.save_image(img, os.path.join('generated_textures', 'progress', f'{obj_name}_{style_name}_res_{width}_step_{step}.png'))
    if step % 100 == 0:
        print("Steps: {}".format(step))

img = image.numpy()
utils.save_image(img, os.path.join('generated_textures', f'{obj_name}_{style_name}_res_{width}.png'))
