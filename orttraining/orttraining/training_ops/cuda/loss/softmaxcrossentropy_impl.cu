// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmaxcrossentropy_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _SoftMaxCrossEntropy(
    const T* log_prob_data,
    const T* label_data,
    CUDA_LONG NORMALIZE_FACTOR,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = -log_prob_data[id] * label_data[id] / NORMALIZE_FACTOR;
}

template <typename T>
void SoftMaxCrossEntropyImpl(
    const T* log_prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG NORMALIZE_FACTOR = static_cast<CUDA_LONG>(normalize_factor);
  _SoftMaxCrossEntropy<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      log_prob,
      label,
      NORMALIZE_FACTOR,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyImpl(T) \
  template void SoftMaxCrossEntropyImpl(       \
      const T* log_prob,                       \
      const T* label,                          \
      size_t normalize_factor,                 \
      T* output_data,                          \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyImpl(float)

    template <typename T>
    __global__ void _SoftMaxCrossEntropyGrad(
        const T* dY,
        const T* log_prob,
        const T* label,
        CUDA_LONG NORMALIZE_FACTOR,
        T* output_data,
        CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = (_Exp(log_prob[id]) - label[id]) * (*dY) / NORMALIZE_FACTOR;
}

template <typename T>
void SoftMaxCrossEntropyGradImpl(
    const T* dY,
    const T* log_prob,
    const T* label,
    size_t normalize_factor,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG NORMALIZE_FACTOR = static_cast<CUDA_LONG>(normalize_factor);
  _SoftMaxCrossEntropyGrad<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      dY,
      log_prob,
      label,
      NORMALIZE_FACTOR,
      output_data,
      N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(T) \
  template void SoftMaxCrossEntropyGradImpl(       \
      const T* dY,                                 \
      const T* log_prob,                           \
      const T* label,                              \
      size_t normalize_factor,                     \
      T* output_data,                              \
      size_t count);

SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(float)

    template <typename T>
    __global__ void _SoftmaxCrossEntropyLoss(
        const T* log_prob_data,
        const T* label_data,
        const T* normalize_factor_data,
        T* output_data,
        CUDA_LONG N,
        CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -log_prob_data[i * D + (int)(label_data[i])] / (*normalize_factor_data);
}

template <typename T>
__global__ void _WeightedSoftmaxCrossEntropyLoss(
    const T* log_prob_data,
    const T* label_data,
    const T* weight_data,
    const T* normalize_factor_data,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  CUDA_KERNEL_ASSERT(label_data[i] >= 0 && label_data[i] < D);
  output_data[i] = -log_prob_data[i * D + (int)(label_data[i])] * weight_data[i] / (*normalize_factor_data);
}

template <typename T>
void SoftmaxCrossEntropyLossImpl(
    const T* log_prob,
    const T* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  if (weight) {
    _WeightedSoftmaxCrossEntropyLoss<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D);
  } else {
    _SoftmaxCrossEntropyLoss<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        log_prob,
        label,
        normalize_factor,
        output_data,
        N,
        D);
  }
}

#define SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(T) \
  template void SoftmaxCrossEntropyLossImpl(       \
      const T* log_prob,                           \
      const T* label,                              \
      const T* weight,                             \
      const T* normalize_factor,                   \
      T* output_data,                              \
      size_t count,                                \
      size_t label_depth);

SPECIALIZED_IMPL_SoftMaxEntropyLossImpl(float)

    template <typename T>
    __global__ void _SoftmaxCrossEntropyLossGrad(
        const T* dY,
        const T* log_prob,
        const T* label,
        const T* normalize_factor,
        T* output_data,
        CUDA_LONG N,
        CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);
  int row = i / D;
  int d = i % D;
  output_data[i] = (*dY) * (_Exp(log_prob[i]) - 1.0 * (d == (int)(label[row]))) / (*normalize_factor);
}

template <typename T>
__global__ void _WeightedSoftmaxCrossEntropyLossGrad(
    const T* dY,
    const T* log_prob,
    const T* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    CUDA_LONG N,
    CUDA_LONG D) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * D);
  int row = i / D;
  int d = i % D;
  output_data[i] = (*dY) * weight[row] * (_Exp(log_prob[i]) - 1.0 * (d == label[row])) / (*normalize_factor);
}

template <typename T>
void SoftmaxCrossEntropyLossGradImpl(
    const T* dY,
    const T* log_prob,
    const T* label,
    const T* weight,
    const T* normalize_factor,
    T* output_data,
    size_t count,
    size_t label_depth) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG D = static_cast<CUDA_LONG>(label_depth);
  int blocksPerGrid = (int)(ceil(static_cast<float>(N * D) / GridDim::maxThreadsPerBlock));
  if (weight) {
    _WeightedSoftmaxCrossEntropyLossGrad<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        log_prob,
        label,
        weight,
        normalize_factor,
        output_data,
        N,
        D);
  } else {
    _SoftmaxCrossEntropyLossGrad<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        dY,
        log_prob,
        label,
        normalize_factor,
        output_data,
        N,
        D);
  }
}

#define SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(T) \
  template void SoftmaxCrossEntropyLossGradImpl(       \
      const T* dY,                                     \
      const T* log_prob,                               \
      const T* label,                                  \
      const T* weight,                                 \
      const T* normalize_factor,                       \
      T* output_data,                                  \
      size_t count,                                    \
      size_t label_depth);

SPECIALIZED_IMPL_SoftMaxEntropyLossGradImpl(float)

}  // namespace cuda
}  // namespace onnxruntime
