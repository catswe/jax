/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

// Unfolds the single dimension in memref.
absl::StatusOr<mlir::Value> MemRefUnfold(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref, int64_t dim,
    const std::vector<std::optional<int64_t>>& factors);

// Slices the memref.
absl::StatusOr<mlir::Value> MemRefSlice(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<std::variant<int64_t, mlir::Value>>& base_indices,
    const std::vector<int64_t>& slice_shape,
    const std::vector<bool>& is_squeezed);

// Transposes the memref.
absl::StatusOr<mlir::Value> MemRefTranspose(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<int64_t>& permutation);

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_
