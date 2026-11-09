// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include <pybind11/pybind11.h>

#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

/**
 * Python bindings for Hf3fsConnector.
 *
 * Exposes the native C++ 3FS connector to Python as LMCacheHf3fsClient.
 * Uses the LMCACHE_BIND_CONNECTOR_METHODS macro to bind standard connector
 * methods.
 *
 * Example usage:
 * @code
 *   from lmcache.lmcache_hf3fs import LMCacheHf3fsClient
 *
 *   client = LMCacheHf3fsClient(
 *       mount_point="/mnt/3fs",
 *       base_paths="/mnt/3fs/path1,/mnt/3fs/path2",
 *       num_workers=8,
 *       ior_entries=256,
 *       io_depth=0,
 *       numa_id=-1,
 *       iov_size=209715200,
 *   )
 * @endcode
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<lmcache::connector::Hf3fsConnector>(m, "LMCacheHf3fsClient",
                                                 py::module_local())
      .def(py::init([](std::string mount_point, std::string base_paths,
                       int num_workers, int ior_entries, int io_depth,
                       int numa_id, size_t iov_size, int time_out,
                       bool enable_key_buffer, py::object per_op_workers,
                       uintptr_t read_buffer, size_t read_buffer_size,
                       py::bytes read_buffer_id, size_t immutable_fd_capacity,
                       bool hash_shards) {
             return new lmcache::connector::Hf3fsConnector(
                 std::move(mount_point), std::move(base_paths), num_workers,
                 ior_entries, io_depth, numa_id, iov_size, time_out,
                 enable_key_buffer,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers),
                 read_buffer, read_buffer_size, std::string(read_buffer_id),
                 immutable_fd_capacity, hash_shards);
           }),
           py::arg("mount_point"), py::arg("base_paths"),
           py::arg("num_workers") = 8, py::arg("ior_entries") = 256,
           py::arg("io_depth") = 0, py::arg("numa_id") = -1,
           py::arg("iov_size") = 209715200, py::arg("time_out") = 200,
           py::arg("enable_key_buffer") = true,
           py::arg("per_op_workers") = py::none(), py::arg("read_buffer") = 0,
           py::arg("read_buffer_size") = 0,
           py::arg("read_buffer_id") = py::bytes(),
           py::arg("immutable_fd_capacity") = 0, py::arg("hash_shards") = false)
      .def("metadata_stats",
           &lmcache::connector::Hf3fsConnector::metadata_stats)
      .def("clear_fd_cache",
           &lmcache::connector::Hf3fsConnector::clear_fd_cache,
           py::call_guard<py::gil_scoped_release>())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::Hf3fsConnector);
}
