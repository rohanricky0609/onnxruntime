// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "orttraining/training_ops/cpu/controlflow/message_queue.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

Status Group::Compute(OpKernelContext* context) const {
  Tensor& output = *context->Output(0, TensorShape({1}));
  bool* output_data = output.template MutableData<bool>();
  *output_data = true;

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Group,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Group);

Status PassThrough::Compute(OpKernelContext* context) const {
  for (int i = 0; i < context->InputCount(); ++i) {
    const auto* X = context->Input<Tensor>(i);
    ORT_ENFORCE(X != nullptr);
    Tensor* Y = context->Output(i, X->Shape());
    ORT_ENFORCE(X->DataRaw() == Y->DataRaw(), "PassThrough input and outpu are not sharing the same buffer.");
  }
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    PassThrough,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .VariadicAlias(0, 0),  // outputs and inputs are mapped one to one
    PassThrough);

Status Yield::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  for (int i_in = 0; i_in < ctx->InputCount(); ++i_in) {
    onnxruntime::contrib::OrtMessageQueue::GetInstance().Push(*ctx_internal->GetInputMLValue(i_in));
  }

  // Reset background event before returning to main thread
  const int64_t background_thread_event_id = 1;
  onnxruntime::contrib::OrtEventPool::GetInstance().ResetEvent(background_thread_event_id);

  // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
  const int64_t main_thread_event_id = 0;
  OrtEventPool::GetInstance().SignalEvent(main_thread_event_id);

  // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
  OrtEventPool::GetInstance().WaitAndResetEvent(background_thread_event_id);

  if (ctx_internal->GetTerminateFlag()) {
    LOGS(ctx->Logger(), WARNING) << "Resumed executing backward subgraph, terminate_flag is set to true.";
  } else {
    // Get output grad from somewhere and prepare Op outputs.
    for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
      ctx_internal->SetOutputMLValue(i_out, OrtMessageQueue::GetInstance().Pop());
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Yield,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .VariadicAlias(0, 0),  // outputs and inputs are mapped one to one
    Yield);

}  // namespace contrib
}  // namespace onnxruntime
