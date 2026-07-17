/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mega_moe_context.h"

#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <hccl/hccl.h>

#include <dlfcn.h>
#include <cstring>
#include <string>

#include <glog/logging.h>

namespace xllm {

namespace {

constexpr uint8_t kCommEngineAiv = 4;
constexpr uint32_t kHcclMaxRankSize = 1024;

struct CommContext {
  uint32_t ep_rank_id = 0;
  uint32_t rank_size_per_server = 0;
  uint64_t kfc_context_addr = 0;
  uint64_t ep_hccl_buffer_[kHcclMaxRankSize] = {};
  uint64_t hcomm_handle_[kHcclMaxRankSize] = {};
};

using HcclKfcAllocOpArgsFn = HcclResult (*)(void**);
using HcclKfcOpArgsSetAlgConfigFn = HcclResult (*)(void*, char*);
using HcclKfcOpArgsSetCommEngineFn = HcclResult (*)(void*, uint8_t);
using HcclCreateOpResCtxFn = HcclResult (*)(HcclComm, uint8_t, void*, void**);
using HcclGetRemoteIpcHcclBufFn = HcclResult (*)(HcclComm, uint64_t, void**, uint64_t*);
using HcclKfcFreeOpArgsFn = HcclResult (*)(void*);
using HcclCommGetHandleWithNameFn = HcclResult (*)(const char*, HcclComm*);
using HcclGetRankSizeFn = HcclResult (*)(HcclComm, uint32_t*);
using HcclGetRankIdFn = HcclResult (*)(HcclComm, uint32_t*);
using HcclGetHcclBufferFn = HcclResult (*)(HcclComm, void**, uint64_t*);

struct KfcFuncTable {
  HcclKfcAllocOpArgsFn alloc_op_args = nullptr;
  HcclKfcOpArgsSetAlgConfigFn set_alg_config = nullptr;
  HcclKfcOpArgsSetCommEngineFn set_comm_engine = nullptr;
  HcclCreateOpResCtxFn create_op_res_ctx = nullptr;
  HcclGetRemoteIpcHcclBufFn get_remote_ipc_hccl_buf = nullptr;
  HcclKfcFreeOpArgsFn free_op_args = nullptr;
  HcclCommGetHandleWithNameFn get_handle_with_name = nullptr;
  HcclGetRankSizeFn get_rank_size = nullptr;
  HcclGetRankIdFn get_rank_id = nullptr;
  HcclGetHcclBufferFn get_hccl_buffer = nullptr;
};

template <typename T>
T load_symbol(void* lib, const char* name) {
  void* sym = dlsym(lib, name);
  CHECK(sym != nullptr) << "dlsym " << name << " failed: " << dlerror();
  return reinterpret_cast<T>(sym);
}

KfcFuncTable load_kfc_functions() {
  void* libhccl = dlopen("libhccl.so", RTLD_NOW | RTLD_GLOBAL);
  CHECK(libhccl != nullptr) << "Failed to load libhccl.so: " << dlerror();
  void* libhccl_fwk = dlopen("libhccl_fwk.so", RTLD_NOW | RTLD_GLOBAL);
  CHECK(libhccl_fwk != nullptr) << "Failed to load libhccl_fwk.so: " << dlerror();

  KfcFuncTable fn;
  fn.alloc_op_args = load_symbol<HcclKfcAllocOpArgsFn>(libhccl, "HcclKfcAllocOpArgs");
  fn.free_op_args = load_symbol<HcclKfcFreeOpArgsFn>(libhccl, "HcclKfcFreeOpArgs");
  fn.set_comm_engine = load_symbol<HcclKfcOpArgsSetCommEngineFn>(libhccl, "HcclKfcOpArgsSetCommEngine");
  fn.set_alg_config = load_symbol<HcclKfcOpArgsSetAlgConfigFn>(libhccl, "HcclKfcOpArgsSetAlgConfig");
  fn.create_op_res_ctx = load_symbol<HcclCreateOpResCtxFn>(libhccl, "HcclCreateOpResCtx");
  fn.get_rank_id = load_symbol<HcclGetRankIdFn>(libhccl, "HcclGetRankId");
  fn.get_rank_size = load_symbol<HcclGetRankSizeFn>(libhccl, "HcclGetRankSize");
  fn.get_hccl_buffer = load_symbol<HcclGetHcclBufferFn>(libhccl, "HcclGetHcclBuffer");
  fn.get_handle_with_name = load_symbol<HcclCommGetHandleWithNameFn>(libhccl_fwk, "HcclCommGetHandleWithName");
  fn.get_remote_ipc_hccl_buf = load_symbol<HcclGetRemoteIpcHcclBufFn>(libhccl_fwk, "HcclGetRemoteIpcHcclBuf");
  return fn;
}

const char* get_soc_name() {
  static const char* soc_name = aclrtGetSocName();
  return soc_name;
}

void collect_rank_buffers(const KfcFuncTable& fn, HcclComm comm,
                          int32_t world_size, int64_t& ccl_buffer_size,
                          CommContext& ctx) {
  uint32_t rank_id = 0;
  HcclResult ret = fn.get_rank_id(comm, &rank_id);
  CHECK(ret == HCCL_SUCCESS) << "HcclGetRankId failed, ret: " << ret;
  ctx.ep_rank_id = rank_id;

  const char* soc_name = get_soc_name();
  if (soc_name != nullptr && strstr(soc_name, "Ascend910B") != nullptr &&
      world_size > 8) {
    return;
  }
  for (int32_t remote_rank = 0; remote_rank < world_size; ++remote_rank) {
    void* remote_addr = nullptr;
    uint64_t comm_size = 0;
    if (static_cast<uint32_t>(remote_rank) == rank_id) {
      ret = fn.get_hccl_buffer(comm, &remote_addr, &comm_size);
      ccl_buffer_size = static_cast<int64_t>(comm_size);
    } else {
      ret = fn.get_remote_ipc_hccl_buf(comm, static_cast<uint64_t>(remote_rank),
                                       &remote_addr, &comm_size);
    }
    CHECK(ret == HCCL_SUCCESS)
        << "Get HcclBufferSize failed for rank " << remote_rank << ", ret: " << ret;
    ctx.ep_hccl_buffer_[remote_rank] = reinterpret_cast<uint64_t>(remote_addr);
  }
}

int64_t context_tensor_size() {
  return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
}

void copy_context_to_tensor(const CommContext& ctx, torch::Tensor& tensor) {
  torch::Tensor host_context = torch::from_blob(
      const_cast<CommContext*>(&ctx),
      {context_tensor_size()}, torch::kInt32);
  tensor.copy_(host_context);
}

}  // namespace

MegaMoeContext create_mega_moe_context(const std::string& group_name,
                                        int32_t world_size) {
  KfcFuncTable fn = load_kfc_functions();

  void* op_args = nullptr;
  HcclResult ret = fn.alloc_op_args(&op_args);
  CHECK(ret == 0) << "HcclKfcAllocOpArgs failed, ret: " << ret;

  ret = fn.set_comm_engine(op_args, kCommEngineAiv);
  CHECK(ret == 0) << "HcclKfcOpArgsSetCommEngine failed, ret: " << ret;

  const char* soc_name = get_soc_name();
  const bool is_910b =
      (soc_name != nullptr && strstr(soc_name, "Ascend910B") != nullptr);
  const bool is_multi_server = world_size > 8;
  const std::string alg_config =
      is_910b && is_multi_server
          ? "BatchWrite=level1:hierarchy"
          : "AlltoAll=level0:fullmesh;level1:pairwise";
  const uint32_t op_type = is_910b && is_multi_server ? 18 : 8;

  ret = fn.set_alg_config(op_args, const_cast<char*>(alg_config.c_str()));
  CHECK(ret == 0) << "HcclKfcOpArgsSetAlgConfig failed, ret: " << ret;

  HcclComm comm_handle = nullptr;
  ret = fn.get_handle_with_name(group_name.c_str(), &comm_handle);
  CHECK(ret == 0) << "HcclCommGetHandleWithName failed for group '"
                  << group_name << "', ret: " << ret;

  void* ops_res_ctx = nullptr;
  ret = fn.create_op_res_ctx(comm_handle, op_type, op_args, &ops_res_ctx);
  CHECK(ret == 0) << "HcclCreateOpResCtx failed, ret: " << ret;

  CommContext ctx = {};
  ctx.kfc_context_addr = reinterpret_cast<uint64_t>(ops_res_ctx);

  uint32_t hccl_world_size = 0;
  ret = fn.get_rank_size(comm_handle, &hccl_world_size);
  CHECK(ret == HCCL_SUCCESS) << "HcclGetRankSize failed, ret: " << ret;
  CHECK(world_size == static_cast<int32_t>(hccl_world_size))
      << "worldSize mismatch: " << world_size << " != " << hccl_world_size;

  ret = fn.free_op_args(op_args);
  CHECK(ret == 0) << "HcclKfcFreeOpArgs failed, ret: " << ret;

  int64_t ccl_buffer_size = 0;
  collect_rank_buffers(fn, comm_handle, world_size, ccl_buffer_size, ctx);

  MegaMoeContext result;
  result.context_tensor = torch::empty(
      {context_tensor_size()},
      torch::TensorOptions().dtype(torch::kInt32)
          .device(c10::DeviceType::PrivateUse1)
          .memory_format(c10::MemoryFormat::Contiguous));
  copy_context_to_tensor(ctx, result.context_tensor);
  result.ccl_buffer_size = ccl_buffer_size;
  return result;
}

}  // namespace xllm
