// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void FillKernel(const Context& dev_ctx,
                const phi::DenseTensor& x UNUSED,
                const phi::Scalar& val,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  T value = val.to<T>();
  const T* value_data = &value;
  cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Fill(
      dev_ctx, pointer_mode, value_data, output_desc.get(), GetBasePtr(out));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fill,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FillKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
