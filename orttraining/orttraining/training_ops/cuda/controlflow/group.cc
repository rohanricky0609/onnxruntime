// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Group,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    onnxruntime::contrib::Group);

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .VariadicAlias(0, 0),  // outputs and inputs are mapped one to one
    onnxruntime::contrib::PassThrough);

ONNX_OPERATOR_KERNEL_EX(
    Yield,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .VariadicAlias(0, 0)  // TODO: this is a hack to avoid allocating output buffer
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    onnxruntime::contrib::Yield);

}  // namespace cuda
}  // namespace onnxruntime
