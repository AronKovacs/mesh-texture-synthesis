import numpy as np
import tensorflow as tf

import os

import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray

import utils

tmscnn_cuda_module_src = """
extern "C"
__global__ void convolution_linear_forward(
    float* targets,
    const int source_filters,
    const int target_filters,
    const float* sources,
    const int* source_indices,
    const float* weights,
    const float* scaling_factors,
    const int n_targets,
    const int terms_per_target) {
    // idx = [target_index, target_index_term, target_filter]

    int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    int target_index_term = blockIdx.y * blockDim.y + threadIdx.y;
    int target_filter = blockIdx.z * blockDim.z + threadIdx.z;

    if (target_index >= n_targets) {
        return;
    }
    if (target_index_term >= terms_per_target) {
        return;
    }
    if (target_filter >= target_filters) {
        return;
    }
    
    int source_index = source_indices[target_index * terms_per_target + target_index_term];

    int weight_id = target_index_term / 4;
    float scaling_factor = scaling_factors[target_index * terms_per_target + target_index_term];

    int weight_id_correction[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    float accum = 0.0;
    for (int j = 0; j < source_filters; j++) {
        int weight_idx = weight_id_correction[weight_id] * target_filters * source_filters + target_filter * source_filters + j;
        accum += sources[source_index * source_filters + j] * weights[weight_idx] * scaling_factor;
    }

    atomicAdd(&targets[target_index * target_filters + target_filter], accum);
}

extern "C"
__global__ void convolution_forward_biases(
    float* targets,
    int n_targets,
    const int target_filters,
    const float* biases) {
    // idx = [target, target_filter]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int target_filter = idx % target_filters;
    idx /= target_filters;
    int target_index = idx;

    if (target_index >= n_targets) {
        return;
    }

    targets[target_index * target_filters + target_filter] += biases[target_filter];
}

extern "C"
__global__ void convolution_linear_backward_sources(
    float* sources_grad,
    const int source_filters,
    const int target_filters,
    const int* source_indices,
    const float* targets_grad,
    const float* weights,
    const float* scaling_factors,
    const int n_targets,
    const int terms_per_target) {
    // idx = [target_index, target_index_term, source_filter]

    int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    int target_index_term = blockIdx.y * blockDim.y + threadIdx.y;
    int source_filter = blockIdx.z * blockDim.z + threadIdx.z;

    if (target_index >= n_targets) {
        return;
    }
    if (target_index_term >= terms_per_target) {
        return;
    }
    if (source_filter >= source_filters) {
        return;
    }

    int source_index = source_indices[target_index * terms_per_target + target_index_term];
    int weight_id = target_index_term / 4;
    float scaling_factor = scaling_factors[target_index * terms_per_target + target_index_term];

    float accum = 0.0;
    for (int i = 0; i < target_filters; i++) {
        int weight_idx = weight_id * target_filters * source_filters + i * source_filters + source_filter;
        accum += targets_grad[target_index * target_filters + i] * weights[weight_idx];
    }
    accum *= scaling_factor;

    float old = atomicAdd(&sources_grad[source_index * source_filters + source_filter], accum);
}

extern "C"
__global__ void convolution_nearest_forward(
    float* targets,
    const int source_filters,
    const int target_filters,
    const float* sources,
    const int* source_indices,
    const float* weights,
    const int n_targets,
    const int terms_per_target) {
    // idx = [target_index, weight_id, target_filter]

    int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    int weight_id = blockIdx.y * blockDim.y + threadIdx.y;
    int target_filter = blockIdx.z * blockDim.z + threadIdx.z;

    if (target_index >= n_targets) {
        return;
    }
    if (weight_id >= 9) {
        return;
    }
    if (target_filter >= target_filters) {
        return;
    }

    int source_index = source_indices[target_index * terms_per_target + weight_id];

    float accum = 0.0;
    for (int j = 0; j < source_filters; j++) {
        int weight_idx = weight_id * target_filters * source_filters + target_filter * source_filters + j;
        accum += sources[source_index * source_filters + j] * weights[weight_idx];
    }

    atomicAdd(&targets[target_index * target_filters + target_filter], accum);
}

extern "C"
__global__ void convolution_nearest_backward_sources(
    float* sources_grad,
    const int source_filters,
    const int target_filters,
    const int* source_indices,
    const float* targets_grad,
    const float* weights,
    const int n_targets,
    const int terms_per_target) {
    // idx = [target_index, weight_id, source_filter]

    int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    int weight_id = blockIdx.y * blockDim.y + threadIdx.y;
    int source_filter = blockIdx.z * blockDim.z + threadIdx.z;

    if (target_index >= n_targets) {
        return;
    }
    if (weight_id >= 9) {
        return;
    }
    if (source_filter >= source_filters) {
        return;
    }

    int source_index = source_indices[target_index * terms_per_target + weight_id];

    float accum = 0.0;
    for (int i = 0; i < target_filters; i++) {
        int weight_idx = weight_id * target_filters * source_filters + i * source_filters + source_filter;
        accum += targets_grad[target_index * target_filters + i] * weights[weight_idx];
    }

    float old = atomicAdd(&sources_grad[source_index * source_filters + source_filter], accum);
}

extern "C"
__global__ void average_pooling_forward(
    float* targets,
    unsigned int n_targets,
    unsigned int n_filters,
    const float* sources,
    const unsigned int* pooling_spans,
    const unsigned int* pooling_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_targets) {
        return;
    }

    unsigned int span_start = pooling_spans[idx];
    unsigned int span_end = pooling_spans[idx + 1];
    float inv_span_length = 1.0f / static_cast<float>(span_end - span_start);

    for (unsigned int filter = 0; filter < n_filters; filter++) {
        float accum = 0.0f;

        for (unsigned int i = span_start; i < span_end; i++) {
            accum += sources[pooling_indices[i] * n_filters + filter];
        }

        targets[idx * n_filters + filter] = accum * inv_span_length;
    }
}

extern "C"
__global__ void average_pooling_backward(
    float* sources_grad,
    unsigned int n_filters,
    const float* targets_grad,
    unsigned int n_targets,
    const unsigned int* pooling_spans,
    const unsigned int* pooling_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_targets) {
        return;
    }

    unsigned int span_start = pooling_spans[idx];
    unsigned int span_end = pooling_spans[idx + 1];
    float inv_span_length = 1.0f / static_cast<float>(span_end - span_start);

    for (unsigned int filter = 0; filter < n_filters; filter++) {
        float gradient = targets_grad[idx * n_filters + filter] * inv_span_length;

        for (unsigned int i = span_start; i < span_end; i++) {
            sources_grad[pooling_indices[i] * n_filters + filter] = gradient;
        }
    }
}

extern "C"
__global__ void zero_int(int* arr, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_size) {
        return;
    }

    arr[idx] = 0;
}

extern "C"
__global__ void zero_unsigned_int(unsigned int* arr, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_size) {
        return;
    }

    arr[idx] = 0;
}

extern "C"
__global__ void zero_float(float* arr, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_size) {
        return;
    }

    arr[idx] = 0.0f;
}

"""

class TmscnnGpu:
    def __init__(self):
        # [mesh][layer]
        self.n_groups = []
        self.n_values = []

        # [mesh][layer]
        self.n_terms = []

        self.current_mesh_ids_in_gpu = None
        self.current_layer_in_gpu = None

        self.forward_cell_ranges_gpu = []
        self.forward_sources_gpu = []
        self.forward_weights_gpu = []
        self.forward_scaling_factors_gpu = []

        self.backward_cell_ranges_gpu = []
        self.backward_targets_gpu = []
        self.backward_weights_gpu = []
        self.backward_scaling_factors_gpu = []

        self.sources_gpu = []
        self.targets_gpu = []
        self.weights_gpu = []
        self.scaling_factors_gpu = []

        self.workspaces_len = {}
        self.workspaces_cpu = {}
        self.workspaces_gpu = {}
        self.streams = {}

        # for each layer this list keeps offsets (in groups) to flattened group representation
        self.flattened_groups_offsets = []

        self.tmscnn_cuda_module = utils.compile_cuda_module(tmscnn_cuda_module_src, no_extern_c=True)

        self.convolution_linear_forward_kernel = self.tmscnn_cuda_module.get_function('convolution_linear_forward')
        self.convolution_linear_backward_sources_kernel = self.tmscnn_cuda_module.get_function('convolution_linear_backward_sources')

        self.convolution_nearest_forward_kernel = self.tmscnn_cuda_module.get_function('convolution_nearest_forward')
        self.convolution_nearest_backward_sources_kernel = self.tmscnn_cuda_module.get_function('convolution_nearest_backward_sources')

        self.convolution_forward_biases_kernel = self.tmscnn_cuda_module.get_function('convolution_forward_biases')

        self.average_pooling_forward_kernel = self.tmscnn_cuda_module.get_function('average_pooling_forward')
        self.average_pooling_backward_kernel = self.tmscnn_cuda_module.get_function('average_pooling_backward')

        self.zero_int_kernel = self.tmscnn_cuda_module.get_function('zero_int')
        self.zero_unsigned_int_kernel = self.tmscnn_cuda_module.get_function('zero_unsigned_int')
        self.zero_float_kernel = self.tmscnn_cuda_module.get_function('zero_float')

    # mesh_ids: list[batch_size]
    # sources_nearest: list[batch_size] of list[n_layers] of np.array((n_groups * 3 * 3), np.float32)
    # sources_linear: list[batch_size] of list[n_layers] of np.array((n_groups * 3 * 3 * 4), np.float32)
    # scaling_factors: list[batch_size] of list[n_layers] of np.array((n_groups * 3 * 3 * 4), np.float32)
    # pooling_spans: list[batch_size] of list[n_layers - 1] of np.array((next_layer_n_groups + 1), np.uint32)
    # pooling_indices: list[batch_size] of list[n_layers - 1] of of np.array((n_groups), np.uint32)
    def use_computation_buffers(self, mesh_ids, sources_nearest, sources_linear, scaling_factors, pooling_spans, pooling_indices, force_rewrite):
        batch_size = len(mesh_ids)
        n_layers = len(sources_nearest[0])

        self.n_groups = [[0 for i in range(n_layers)] for j in range(batch_size)]
        self.flattened_groups_offsets = [[0 for i in range(batch_size + 1)] for j in range(n_layers)]

        for i in range(batch_size):
            mesh_id = mesh_ids[i]
            for layer in range(n_layers):
                mesh_layer_id = f'{mesh_id}_{layer}'
                n_groups = sources_nearest[i][layer].shape[0] // 9

                self.n_groups[i][layer] = n_groups
                self.flattened_groups_offsets[layer][i + 1] = self.flattened_groups_offsets[layer][i] + n_groups

                if not force_rewrite and self.workspace_exists(f'terms_sources_linear_{mesh_layer_id}'):
                    continue

                self.allocate_workspace(f'terms_sources_linear_{mesh_layer_id}', n_groups * 3 * 3 * 4, np.int32)
                self.allocate_workspace(f'terms_scaling_factors_linear_{mesh_layer_id}', n_groups * 3 * 3 * 4, np.float32)

                self.workspace_htod(f'terms_sources_linear_{mesh_layer_id}', sources_linear[i][layer])
                self.workspace_htod(f'terms_scaling_factors_linear_{mesh_layer_id}', scaling_factors[i][layer])

                self.allocate_workspace(f'terms_sources_nearest_{mesh_layer_id}', n_groups * 3 * 3, np.int32)
                self.workspace_htod(f'terms_sources_nearest_{mesh_layer_id}', sources_nearest[i][layer])

                if layer < n_layers - 1:
                    self.allocate_workspace(f'pooling_spans_{mesh_layer_id}', pooling_spans[i][layer].shape[0], np.uint32)
                    self.allocate_workspace(f'pooling_indices_{mesh_layer_id}', pooling_indices[i][layer].shape[0], np.uint32)

                    self.workspace_htod(f'pooling_spans_{mesh_layer_id}', pooling_spans[i][layer])
                    self.workspace_htod(f'pooling_indices_{mesh_layer_id}', pooling_indices[i][layer])

    def allocate_workspace(self, id, n, dtype):
        if dtype != np.int32 and dtype != np.uint32 and dtype != np.float32:
            raise Exception('dtype must be np.int32 or np.uint32 or np.float32')
        self.workspaces_len[id] = n
        if id in self.workspaces_cpu:
            if self.workspaces_cpu[id].shape[0] >= n and self.workspaces_cpu[id].dtype == dtype:
                return
            del self.workspaces_cpu[id]
            del self.workspaces_gpu[id]
        self.workspaces_cpu[id] = cuda.pagelocked_zeros(n, dtype=dtype)
        self.workspaces_gpu[id] = GPUArray(n, dtype=dtype)

    def zero_workspace(self, id, cpu = True, gpu = True, stream = None):
        ws_cpu = self.workspaces_cpu[id]
        ws_gpu = self.workspaces_gpu[id]
        
        block_size = 1024
        grid_size = (self.workspaces_len[id] + block_size - 1) // block_size

        if ws_cpu.dtype == np.int32:
            if cpu:
                ws_cpu.fill(0)
            if gpu:
                self.zero_int_kernel(
                    ws_gpu,
                    np.int32(self.workspaces_len[id]),
                    block = (block_size, 1, 1),
                    grid = (grid_size, 1, 1),
                    stream = stream
                )
        elif ws_cpu.dtype == np.uint32:
            if cpu:
                ws_cpu.fill(0)
            if gpu:
                self.zero_unsigned_int_kernel(
                    ws_gpu,
                    np.int32(self.workspaces_len[id]),
                    block = (block_size, 1, 1),
                    grid = (grid_size, 1, 1),
                    stream = stream
                )
        elif ws_cpu.dtype == np.float32:
            if cpu:
                ws_cpu.fill(0.0)
            if gpu:
                self.zero_float_kernel(
                    ws_gpu,
                    np.int32(self.workspaces_len[id]),
                    block = (block_size, 1, 1),
                    grid = (grid_size, 1, 1),
                    stream = stream
                )
        else:
            raise Exception('dtype must be np.int32 or np.float32')

    def workspace_htod(self, id, data = None, stream = None):
        length = self.workspaces_len[id]
        if data is not None:
            self.workspaces_cpu[id][:length] = data.flatten()[:length]
        if stream is None:
            cuda.memcpy_htod(self.workspaces_gpu[id][:length].ptr, self.workspaces_cpu[id][:length])
        else:
            cuda.memcpy_htod_async(self.workspaces_gpu[id][:length].ptr, self.workspaces_cpu[id][:length], stream=stream)

    def workspace_dtoh(self, id, stream = None):
        length = self.workspaces_len[id]
        if stream is None:
            cuda.memcpy_dtoh(self.workspaces_cpu[id][:length], self.workspaces_gpu[id][:length].ptr)
        else:
            cuda.memcpy_dtoh_async(self.workspaces_cpu[id][:length], self.workspaces_gpu[id][:length].ptr, stream=stream)

    def workspace_exists(self, id):
        try:
            self.workspaces_len[id]
            return True
        except:
            return False

    def stream(self, id):
        if id in self.streams:
            return self.streams[id]
        self.streams[id] = cuda.Stream()
        return self.streams[id]

class TMSCNNConvolutionGPU(tf.keras.layers.Layer):
    def __init__(self, tmscnn, tmscnn_gpu, layer, filters, kernel_size, interpolation, activation=None, kernel_initializer='ones', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, name=None, **kwargs):
        super(TMSCNNConvolutionGPU, self).__init__(name=name, dynamic=True, **kwargs)
        self.tmscnn = tmscnn
        self.tmscnn_gpu = tmscnn_gpu
        if interpolation == 'linear':
            self.terms_per_target = 3 * 3 * 4
        elif interpolation == 'nearest':
            self.terms_per_target = 3 * 3
        else:
            raise Exception(f'unsupported interpolation {interpolation}')
        self.interpolation = interpolation

        if activation is None:
            self.activation_function = None
            self.activation_args = None
        elif type(activation) is tuple or type(activation) is list:
            self.activation_function = activation[0]
            self.activation_args = activation[1]
        elif type(activation) is str:
            self.activation_function = activation
            self.activation_args = None
        else:
            raise Exception(f'unknown activation format: {activation}')
        
        if self.activation_function is not None and self.activation_function != 'relu':
            raise Exception(f'unknown actication function')

        self.layer = layer
        self.target_filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.input_is_gpu_cooperative = False
        self.output_is_gpu_cooperative = False

        self.first_call = True

    def build(self, input_shape):
        self.source_filters = input_shape[1][-1]
        with tf.device('/device:CPU:0'):
            self.w = self.add_weight(
                shape=(self.kernel_size[0] * self.kernel_size[1], self.target_filters, self.source_filters),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                dtype=tf.float32
            )
            self.b = self.add_weight(
                shape=(self.target_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=tf.float32
            )

        self.w_gpu = GPUArray(self.kernel_size[0] * self.kernel_size[1] * self.target_filters * self.source_filters, dtype=np.float32)
        self.b_gpu = GPUArray(self.target_filters, dtype=np.float32)

    def call(self, inputs):
        @tf.custom_gradient
        def convolution(mesh_ids, sources, weights, biases):
            def convolution_grad(dy):
                dy_numpy = None
                if not self.output_is_gpu_cooperative:
                    dy_numpy = dy.numpy()
                if self.activation_function == 'relu':
                    dy_numpy = dy_numpy * (result_before_relu > 0)

                for i in range(batch_size):
                    stream_id = f'{i}'
                    stream = self.tmscnn_gpu.stream(stream_id)

                    mesh_layer_id = f'{mesh_ids[0][i]}_{self.layer}'

                    current_n_groups = self.tmscnn_gpu.n_groups[i][self.layer]

                    sources_workspace_length = self.source_filters * current_n_groups
                    targets_workspace_length = self.target_filters * current_n_groups

                    if self.output_is_gpu_cooperative:
                        targets_grad_workspace_name = self.output_layer_backward_output_name(i)
                    else:
                        targets_grad_workspace_name = f'targets_grad_{stream_id}'
                        self.tmscnn_gpu.allocate_workspace(targets_grad_workspace_name, targets_workspace_length, np.float32)

                        self.tmscnn_gpu.workspace_htod(
                            targets_grad_workspace_name,
                            data=dy_numpy.flatten()[
                                self.tmscnn_gpu.flattened_groups_offsets[self.layer][i] * self.target_filters:
                                self.tmscnn_gpu.flattened_groups_offsets[self.layer][i + 1] * self.target_filters],
                            stream=stream)

                    block_size = 512
                    grid_size = (targets_workspace_length + block_size - 1) // block_size

                    if self.activation_function == 'sigmoid':
                        self.tmscnn_gpu.sigmoid_backward(
                            self.tmscnn_gpu.workspaces_gpu[targets_grad_workspace_name],
                            np.int32(targets_workspace_length),

                            block = (block_size, 1, 1),
                            grid = (grid_size, 1, 1)
                        )

                    sources_grad_workspace_name = f"backward_output_{i}"
                    self.tmscnn_gpu.allocate_workspace(sources_grad_workspace_name, sources_workspace_length, np.float32)

                    self.tmscnn_gpu.zero_workspace(sources_grad_workspace_name, cpu=False, stream=stream)

                    block_size_target_index = 4
                    block_size_target_index_term = 4
                    block_size_source_filter = 32

                    grid_size_target_index = int((current_n_groups + block_size_target_index - 1) // block_size_target_index)
                    grid_size_target_index_term = int((self.terms_per_target + block_size_target_index_term - 1) // block_size_target_index_term)
                    grid_size_source_filter = int((self.source_filters + block_size_target_filter - 1) // block_size_target_filter)

                    if self.interpolation == 'linear':
                        self.tmscnn_gpu.convolution_linear_backward_sources_kernel(
                            self.tmscnn_gpu.workspaces_gpu[sources_grad_workspace_name],
                            np.int32(self.source_filters),
                            np.int32(self.target_filters),
                            self.tmscnn_gpu.workspaces_gpu[f'terms_sources_{self.interpolation}_{mesh_layer_id}'],
                            self.tmscnn_gpu.workspaces_gpu[targets_grad_workspace_name],
                            self.w_gpu,
                            self.tmscnn_gpu.workspaces_gpu[f'terms_scaling_factors_{self.interpolation}_{mesh_layer_id}'],
                            np.int32(current_n_groups),
                            np.int32(self.terms_per_target),

                            block = (block_size_target_index, block_size_target_index_term, block_size_source_filter),
                            grid = (grid_size_target_index, grid_size_target_index_term, grid_size_source_filter),
                            stream = stream
                        )
                    elif self.interpolation == 'nearest':
                        self.tmscnn_gpu.convolution_nearest_backward_sources_kernel(
                            self.tmscnn_gpu.workspaces_gpu[sources_grad_workspace_name],
                            np.int32(self.source_filters),
                            np.int32(self.target_filters),
                            self.tmscnn_gpu.workspaces_gpu[f'terms_sources_{self.interpolation}_{mesh_layer_id}'],
                            self.tmscnn_gpu.workspaces_gpu[targets_grad_workspace_name],
                            self.w_gpu,
                            np.int32(current_n_groups),
                            np.int32(self.terms_per_target),

                            block = (block_size_target_index, block_size_target_index_term, block_size_source_filter),
                            grid = (grid_size_target_index, grid_size_target_index_term, grid_size_source_filter),
                            stream = stream
                        )

                    if not self.input_is_gpu_cooperative:
                        self.tmscnn_gpu.workspace_dtoh(sources_grad_workspace_name, stream=stream)

                if self.input_is_gpu_cooperative:
                    for i in range(batch_size):
                        stream_id = f'{i}'
                        stream = self.tmscnn_gpu.stream(stream_id)
                        stream.synchronize()

                        sources_grad_workspace_name = self.backward_output_name(i)
                    
                    with tf.device('/device:CPU:0'):
                        sources_grad = tf.zeros((1, self.source_filters), dtype=tf.float32)
                else:
                    sources_grad = np.empty(sources_shape, dtype=np.float32)

                    for i in range(batch_size):
                        stream_id = f'{i}'
                        stream = self.tmscnn_gpu.stream(stream_id)
                        stream.synchronize()

                        flattened_start = self.tmscnn_gpu.flattened_groups_offsets[self.layer][i]
                        flattened_end = self.tmscnn_gpu.flattened_groups_offsets[self.layer][i + 1]

                        sources_grad_workspace_name = f"backward_output_{i}"
                        partial_sources_grad = self.tmscnn_gpu.workspaces_cpu[sources_grad_workspace_name][:self.tmscnn_gpu.workspaces_len[sources_grad_workspace_name]]
                        partial_sources_grad = np.reshape(partial_sources_grad, (partial_sources_grad.shape[0] // self.source_filters, self.source_filters))

                        sources_grad[flattened_start:flattened_end] = partial_sources_grad

                    with tf.device('/device:CPU:0'):
                        sources_grad = tf.convert_to_tensor(np.reshape(np.copy(sources_grad), sources_shape))

                with tf.device('/device:CPU:0'):
                    weights_grad = tf.convert_to_tensor(np.zeros(self.w.get_shape(), dtype=np.float32))
                    biases_grad = tf.convert_to_tensor(np.zeros(self.b.get_shape(), dtype=np.float32))

                return (tf.zeros(mesh_ids.get_shape(), dtype=tf.int32), sources_grad, weights_grad, biases_grad)
            batch_size = mesh_ids.get_shape()[0]
            sources_numpy = None
            if not self.input_is_gpu_cooperative:
                sources_numpy = sources.numpy()
            sources_shape = (self.tmscnn_gpu.flattened_groups_offsets[self.layer][-1], self.source_filters)

            self.transfer_wb_to_gpu()

            for i in range(batch_size):
                stream_id = f'{i}'
                stream = self.tmscnn_gpu.stream(stream_id)

                mesh_layer_id = f'{mesh_ids[0][i]}_{self.layer}'

                current_n_groups = self.tmscnn_gpu.n_groups[i][self.layer]

                sources_workspace_length = self.source_filters * current_n_groups
                targets_workspace_length = self.target_filters * current_n_groups

                if self.input_is_gpu_cooperative:
                    sources_workspace_name = self.input_layer_forward_output_name(i)
                else:
                    sources_workspace_name = f"input_sources_{stream_id}"
                    self.tmscnn_gpu.allocate_workspace(sources_workspace_name, sources_workspace_length, np.float32)
                    self.tmscnn_gpu.workspace_htod(
                        sources_workspace_name,
                        data=sources_numpy.flatten()[
                            self.tmscnn_gpu.flattened_groups_offsets[self.layer][i] * self.source_filters:
                            self.tmscnn_gpu.flattened_groups_offsets[self.layer][i + 1] * self.source_filters],
                        stream=stream)

                targets_workspace_name = f"forward_output_{i}"
                self.tmscnn_gpu.allocate_workspace(targets_workspace_name, targets_workspace_length, np.float32)
                self.tmscnn_gpu.zero_workspace(targets_workspace_name, cpu=False)

                block_size_target_index = 4
                block_size_target_index_term = 4
                block_size_target_filter = 32

                grid_size_target_index = int((current_n_groups + block_size_target_index - 1) // block_size_target_index)
                grid_size_target_index_term = int((self.terms_per_target + block_size_target_index_term - 1) // block_size_target_index_term)
                grid_size_target_filter = int((self.target_filters + block_size_target_filter - 1) // block_size_target_filter)

                if self.interpolation == 'linear':
                    self.tmscnn_gpu.convolution_linear_forward_kernel(
                        self.tmscnn_gpu.workspaces_gpu[targets_workspace_name],
                        np.int32(self.source_filters),
                        np.int32(self.target_filters),
                        self.tmscnn_gpu.workspaces_gpu[sources_workspace_name],
                        self.tmscnn_gpu.workspaces_gpu[f'terms_sources_{self.interpolation}_{mesh_layer_id}'],
                        self.w_gpu,
                        self.tmscnn_gpu.workspaces_gpu[f'terms_scaling_factors_{self.interpolation}_{mesh_layer_id}'],
                        np.int32(current_n_groups),
                        np.int32(self.terms_per_target),

                        block = (block_size_target_index, block_size_target_index_term, block_size_target_filter),
                        grid = (grid_size_target_index, grid_size_target_index_term, grid_size_target_filter),
                        stream = stream
                    )
                elif self.interpolation == 'nearest':
                    self.tmscnn_gpu.convolution_nearest_forward_kernel(
                        self.tmscnn_gpu.workspaces_gpu[targets_workspace_name],
                        np.int32(self.source_filters),
                        np.int32(self.target_filters),
                        self.tmscnn_gpu.workspaces_gpu[sources_workspace_name],
                        self.tmscnn_gpu.workspaces_gpu[f'terms_sources_{self.interpolation}_{mesh_layer_id}'],
                        self.w_gpu,
                        np.int32(current_n_groups),
                        np.int32(self.terms_per_target),

                        block = (block_size_target_index, block_size_target_index_term, block_size_target_filter),
                        grid = (grid_size_target_index, grid_size_target_index_term, grid_size_target_filter),
                        stream = stream
                    )

                block_size = 256
                grid_size = int(((current_n_groups * self.target_filters) + block_size - 1) // block_size)

                self.tmscnn_gpu.convolution_forward_biases_kernel(
                    self.tmscnn_gpu.workspaces_gpu[targets_workspace_name],
                    np.int32(current_n_groups),
                    np.int32(self.target_filters),
                    self.b_gpu,

                    block = (block_size, 1, 1),
                    grid = (grid_size, 1, 1),
                    stream = stream
                )

                block_size = 512
                grid_size = (targets_workspace_length + block_size - 1) // block_size

                if not self.output_is_gpu_cooperative:
                    self.tmscnn_gpu.workspace_dtoh(targets_workspace_name, stream=stream)            

            if self.output_is_gpu_cooperative:
                for i in range(batch_size):
                    stream_id = f'{i}'
                    self.tmscnn_gpu.stream(stream_id).synchronize()
                result_tensor = tf.zeros((1, self.target_filters))
            else:
                result = np.zeros((sources_shape[0], self.target_filters), dtype=np.float32)

                for i in range(batch_size):
                    stream_id = f'{i}'
                    self.tmscnn_gpu.stream(stream_id).synchronize()

                    targets_workspace_name = f"forward_output_{i}"

                    flattened_start = self.tmscnn_gpu.flattened_groups_offsets[self.layer][i]
                    flattened_end = self.tmscnn_gpu.flattened_groups_offsets[self.layer][i + 1]

                    partial_result = self.tmscnn_gpu.workspaces_cpu[targets_workspace_name][:self.tmscnn_gpu.workspaces_len[targets_workspace_name]]
                    partial_result = np.reshape(partial_result, (partial_result.shape[0] // self.target_filters, self.target_filters))

                    result[flattened_start:flattened_end] = partial_result

                if self.activation_function == 'relu':
                    result_before_relu = np.copy(result)
                    result = result * (result > 0)

                result_tensor = tf.convert_to_tensor(np.reshape(result, (sources_shape[0], self.target_filters)))

            return result_tensor, convolution_grad

        with tf.device('/device:CPU:0'):
            if self.first_call:
                self.first_call = False
                return tf.zeros((1, self.target_filters))
            result = convolution(inputs[0], inputs[1], self.w, self.b)
        return result

    def is_gpu_cooperative(self):
        return True

    def set_input_layer(self, input_layer):
        self.input_layer_name = input_layer.name
        self.input_is_gpu_cooperative = False
        try:
            self.input_is_gpu_cooperative = input_layer.is_gpu_cooperative()
        except:
            pass

    def set_output_layer(self, output_layer):
        self.output_layer_name = output_layer.name
        self.output_is_gpu_cooperative = False
        try:
            self.output_is_gpu_cooperative = output_layer.is_gpu_cooperative()
        except:
            pass

    def forward_output_name(self, batch_element):
        return f'{self.name}_forward_output_{batch_element}'

    def backward_output_name(self, batch_element):
        return f'{self.name}_backward_output_{batch_element}'

    def input_layer_forward_output_name(self, batch_element):
        return f'{self.input_layer_name}_forward_output_{batch_element}'

    def output_layer_backward_output_name(self, batch_element):
        return f'{self.output_layer_name}_backward_output_{batch_element}'

    def transfer_wb_to_gpu(self):
        cuda.memcpy_htod(self.w_gpu.ptr, self.w.numpy().flatten())
        cuda.memcpy_htod(self.b_gpu.ptr, self.b.numpy().flatten())

    def write_weights_to_dir(self, dir):
        w_path = os.path.join(dir, self.name + '_w')
        b_path = os.path.join(dir, self.name + '_b')
        utils.save(self.w.numpy(), w_path)
        utils.save(self.b.numpy(), b_path)

    def load_weights_from_dir(self, dir):
        w_path = os.path.join(dir, self.name + '_w')
        b_path = os.path.join(dir, self.name + '_b')
        with tf.device('/device:CPU:0'):
            self.w.assign(tf.convert_to_tensor(utils.load(w_path)))
            self.b.assign(tf.convert_to_tensor(utils.load(b_path)))

class TMSCNNAveragePoolingGPU(tf.keras.layers.Layer):
    def __init__(self, tmscnn, tmscnn_gpu, source_layer, **kwargs):
        super(TMSCNNAveragePoolingGPU, self).__init__(dynamic=True, **kwargs)
        self.tmscnn = tmscnn
        self.tmscnn_gpu = tmscnn_gpu
        self.source_layer = source_layer

        self.input_is_gpu_cooperative = False
        self.output_is_gpu_cooperative = False

        self.first_call = True
    
    def build(self, input_shape):
        self.n_filters = input_shape[1][-1]

    def call(self, inputs):
        @tf.custom_gradient
        def average_pooling(mesh_ids, sources):
            def average_pooling_grad(dy):
                dy_numpy = None
                if not self.output_is_gpu_cooperative:
                    dy_numpy = dy.numpy()

                for i in range(batch_size):
                    stream_id = f'{i}'
                    stream = self.tmscnn_gpu.stream(stream_id)

                    mesh_layer_id = f'{mesh_ids[0][i]}_{self.source_layer}'

                    source_n_groups = self.tmscnn_gpu.n_groups[i][self.source_layer]
                    target_n_groups = self.tmscnn_gpu.n_groups[i][self.source_layer + 1]

                    sources_workspace_length = self.n_filters * source_n_groups
                    targets_workspace_length = self.n_filters * target_n_groups

                    if self.output_is_gpu_cooperative:
                        targets_grad_workspace_name = self.output_layer_backward_output_name(i)
                    else:
                        targets_grad_workspace_name = f'targets_grad_{stream_id}'
                        self.tmscnn_gpu.allocate_workspace(targets_grad_workspace_name, targets_workspace_length, np.float32)

                        self.tmscnn_gpu.workspace_htod(
                            targets_grad_workspace_name,
                            data=dy_numpy.flatten()[
                                self.tmscnn_gpu.flattened_groups_offsets[self.source_layer + 1][i] * self.n_filters:
                                self.tmscnn_gpu.flattened_groups_offsets[self.source_layer + 1][i + 1] * self.n_filters],
                            stream=stream)

                    block_size = 512
                    grid_size = (targets_workspace_length + block_size - 1) // block_size

                    sources_grad_workspace_name = f"backward_output_{i}"
                    self.tmscnn_gpu.allocate_workspace(sources_grad_workspace_name, sources_workspace_length, np.float32)
                    self.tmscnn_gpu.zero_workspace(sources_grad_workspace_name)

                    block_size = 32
                    grid_size = (target_n_groups + block_size - 1) // block_size

                    self.tmscnn_gpu.average_pooling_backward_kernel(
                        self.tmscnn_gpu.workspaces_gpu[sources_grad_workspace_name],
                        np.uint32(self.n_filters),
                        self.tmscnn_gpu.workspaces_gpu[targets_grad_workspace_name],
                        np.uint32(target_n_groups),
                        self.tmscnn_gpu.workspaces_gpu[f'pooling_spans_{mesh_layer_id}'],
                        self.tmscnn_gpu.workspaces_gpu[f'pooling_indices_{mesh_layer_id}'],

                        block = (block_size, 1, 1),
                        grid = (grid_size, 1, 1),
                        stream = stream
                    )

                    if not self.input_is_gpu_cooperative:
                        self.tmscnn_gpu.workspace_dtoh(sources_grad_workspace_name, stream=stream)

                if self.input_is_gpu_cooperative:
                    for i in range(batch_size):
                        stream_id = f'{i}'
                        stream = self.tmscnn_gpu.stream(stream_id)
                        stream.synchronize()

                        sources_grad_workspace_name = self.backward_output_name(i)
                    
                    with tf.device('/device:CPU:0'):
                        sources_grad = tf.zeros((1, self.n_filters), dtype=tf.float32)
                else:
                    sources_grad = np.empty(sources_shape, dtype=np.float32)

                    for i in range(batch_size):
                        stream_id = f'{i}'
                        stream = self.tmscnn_gpu.stream(stream_id)
                        stream.synchronize()

                        flattened_start = self.tmscnn_gpu.flattened_groups_offsets[self.source_layer][i]
                        flattened_end = self.tmscnn_gpu.flattened_groups_offsets[self.source_layer][i + 1]

                        sources_grad_workspace_name = f"backward_output_{i}"
                        partial_sources_grad = self.tmscnn_gpu.workspaces_cpu[sources_grad_workspace_name][:self.tmscnn_gpu.workspaces_len[sources_grad_workspace_name]]
                        partial_sources_grad = np.reshape(partial_sources_grad, (partial_sources_grad.shape[0] // self.n_filters, self.n_filters))

                        sources_grad[flattened_start:flattened_end] = partial_sources_grad

                    with tf.device('/device:CPU:0'):
                        sources_grad = tf.convert_to_tensor(np.reshape(np.copy(sources_grad), sources_shape))

                return (tf.zeros(mesh_ids.get_shape(), dtype=tf.int32), sources_grad)
            batch_size = mesh_ids.get_shape()[0]
            sources_numpy = None
            if not self.input_is_gpu_cooperative:
                sources_numpy = sources.numpy()
            sources_shape = (self.tmscnn_gpu.flattened_groups_offsets[self.source_layer][-1], self.n_filters)
            targets_shape = (self.tmscnn_gpu.flattened_groups_offsets[self.source_layer + 1][-1], self.n_filters)

            for i in range(batch_size):
                stream_id = f'{i}'
                stream = self.tmscnn_gpu.stream(stream_id)

                mesh_layer_id = f'{mesh_ids[0][i]}_{self.source_layer}'

                source_n_groups = self.tmscnn_gpu.n_groups[i][self.source_layer]
                target_n_groups = self.tmscnn_gpu.n_groups[i][self.source_layer + 1]

                sources_workspace_length = self.n_filters * source_n_groups
                targets_workspace_length = self.n_filters * target_n_groups

                if self.input_is_gpu_cooperative:
                    sources_workspace_name = self.input_layer_forward_output_name(i)
                else:
                    sources_workspace_name = f'input_sources_{stream_id}'
                    self.tmscnn_gpu.allocate_workspace(sources_workspace_name, sources_workspace_length, np.float32)
                    self.tmscnn_gpu.workspace_htod(
                        sources_workspace_name,
                        data=sources_numpy.flatten()[
                            self.tmscnn_gpu.flattened_groups_offsets[self.source_layer][i] * self.n_filters:
                            self.tmscnn_gpu.flattened_groups_offsets[self.source_layer][i + 1] * self.n_filters],
                        stream=stream)

                targets_workspace_name = f"forward_output_{i}"
                self.tmscnn_gpu.allocate_workspace(targets_workspace_name, targets_workspace_length, np.float32)

                block_size = 32
                grid_size = (target_n_groups + block_size - 1) // block_size

                self.tmscnn_gpu.average_pooling_forward_kernel(
                    self.tmscnn_gpu.workspaces_gpu[targets_workspace_name],
                    np.uint32(target_n_groups),
                    np.uint32(self.n_filters),
                    self.tmscnn_gpu.workspaces_gpu[sources_workspace_name],
                    self.tmscnn_gpu.workspaces_gpu[f'pooling_spans_{mesh_layer_id}'],
                    self.tmscnn_gpu.workspaces_gpu[f'pooling_indices_{mesh_layer_id}'],

                    block = (block_size, 1, 1),
                    grid = (grid_size, 1, 1),
                    stream = stream
                )

                block_size = 512
                grid_size = (targets_workspace_length + block_size - 1) // block_size

                if not self.output_is_gpu_cooperative:
                    self.tmscnn_gpu.workspace_dtoh(targets_workspace_name, stream=stream)            

            if self.output_is_gpu_cooperative:
                for i in range(batch_size):
                    stream_id = f'{i}'
                    self.tmscnn_gpu.stream(stream_id).synchronize()
                result_tensor = tf.zeros((1, self.n_filters))
            else:
                result = np.zeros(targets_shape, dtype=np.float32)

                for i in range(batch_size):
                    stream_id = f'{i}'
                    self.tmscnn_gpu.stream(stream_id).synchronize()

                    targets_workspace_name = f"forward_output_{i}"

                    flattened_start = self.tmscnn_gpu.flattened_groups_offsets[self.source_layer + 1][i]
                    flattened_end = self.tmscnn_gpu.flattened_groups_offsets[self.source_layer + 1][i + 1]

                    partial_result = self.tmscnn_gpu.workspaces_cpu[targets_workspace_name][:self.tmscnn_gpu.workspaces_len[targets_workspace_name]]
                    partial_result = np.reshape(partial_result, (partial_result.shape[0] // self.n_filters, self.n_filters))

                    result[flattened_start:flattened_end] = partial_result

                result_tensor = tf.convert_to_tensor(np.reshape(result, targets_shape))

            return result_tensor, average_pooling_grad

        with tf.device('/device:CPU:0'):
            if self.first_call:
                self.first_call = False
                return tf.zeros((1, self.n_filters))
            result = average_pooling(inputs[0], inputs[1])
        return result

    def is_gpu_cooperative(self):
        return True

    def set_input_layer(self, input_layer):
        self.input_layer_name = input_layer.name
        self.input_is_gpu_cooperative = False
        try:
            self.input_is_gpu_cooperative = input_layer.is_gpu_cooperative()
        except:
            pass

    def set_output_layer(self, output_layer):
        self.output_layer_name = output_layer.name
        self.output_is_gpu_cooperative = False
        try:
            self.output_is_gpu_cooperative = output_layer.is_gpu_cooperative()
        except:
            pass

    def forward_output_name(self, batch_element):
        return f'{self.name}_forward_output_{batch_element}'

    def backward_output_name(self, batch_element):
        return f'{self.name}_backward_output_{batch_element}'

    def input_layer_forward_output_name(self, batch_element):
        return f'{self.input_layer_name}_forward_output_{batch_element}'

    def output_layer_backward_output_name(self, batch_element):
        return f'{self.output_layer_name}_backward_output_{batch_element}'

class TMSCNNTexturesToFlattenedGroups(tf.keras.layers.Layer):
    def __init__(self, tmscnn, **kwargs):
        super(TMSCNNTexturesToFlattenedGroups, self).__init__(dynamic=True, **kwargs)
        self.tmscnn = tmscnn

    def call(self, inputs):
        @tf.custom_gradient
        def textures_to_flattened_groups(mesh_ids, textures):
            def textures_to_flattened_groups_grad(dy):
                textures_grad_numpy = np.zeros(textures.get_shape(), dtype=np.float32)
                dy_numpy = dy.numpy()
                self.tmscnn.textures_to_flattened_groups_backward_pass(textures_grad_numpy, mesh_ids_numpy, dy_numpy)
                with tf.device('/device:CPU:0'):
                    textures_grad = tf.convert_to_tensor(textures_grad_numpy)

                return (tf.zeros(mesh_ids.get_shape(), dtype=tf.int32), textures_grad)
            
            mesh_ids_numpy = mesh_ids.numpy()
            textures_numpy = textures.numpy()

            flattened_groups = self.tmscnn.textures_to_flattened_groups_forward_pass(mesh_ids_numpy, textures_numpy)
            flattened_groups_tensor = tf.convert_to_tensor(flattened_groups)

            return flattened_groups_tensor, textures_to_flattened_groups_grad

        with tf.device('/device:CPU:0'):
            result = textures_to_flattened_groups(inputs[0], inputs[1])
        return result

class TMSCNNAveragePooling(tf.keras.layers.Layer):
    def __init__(self, tmscnn, source_layer, **kwargs):
        super(TMSCNNAveragePooling, self).__init__(dynamic=True, **kwargs)
        self.tmscnn = tmscnn
        self.source_layer = source_layer

    def call(self, inputs):
        @tf.custom_gradient
        def average_pooling(mesh_ids, sources):
            def average_pooling_grad(dy):
                sources_grad_numpy = np.zeros(sources.get_shape(), dtype=np.float32)
                dy_numpy = dy.numpy()
                self.tmscnn.average_pooling_backward_pass(self.source_layer, sources_grad_numpy, mesh_ids_numpy, dy_numpy)
                with tf.device('/device:CPU:0'):
                    sources_grad = tf.convert_to_tensor(sources_grad_numpy)
                return (tf.zeros(mesh_ids.get_shape(), dtype=tf.int32), sources_grad)

            mesh_ids_numpy = mesh_ids.numpy()
            sources_numpy = sources.numpy()

            result = self.tmscnn.average_pooling_forward_pass(self.source_layer, mesh_ids_numpy, sources_numpy)

            result_tensor = tf.convert_to_tensor(result)

            return result_tensor, average_pooling_grad

        with tf.device('/device:CPU:0'):
            result = average_pooling(inputs[0], inputs[1])
        return result