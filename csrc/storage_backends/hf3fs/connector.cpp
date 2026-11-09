// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include "connector.h"

#include <limits.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "../keys.h"

namespace lmcache {
namespace connector {

/**
 * Construct a new Hf3fsConnector.
 *
 * Validates:
 * - mount_point is a valid 3FS path (via hf3fs_extract_mount_point)
 * - Each base_path is a subdirectory of mount_point
 * - Parameters are within valid ranges
 */
Hf3fsConnector::Hf3fsConnector(std::string mount_point, std::string base_paths,
                               int num_workers, int ior_entries, int io_depth,
                               int numa_id, size_t iov_size, int time_out,
                               bool enable_key_buffer,
                               WorkerPoolConfig worker_pool_config,
                               uintptr_t read_buffer, size_t read_buffer_size,
                               std::string read_buffer_id,
                               size_t immutable_fd_capacity, bool hash_shards)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      metadata_metrics_(std::make_shared<MetadataMetrics>()),
      read_files_(immutable_fd_capacity, metadata_metrics_),
      hash_shards_(hash_shards),
      mount_point_(std::move(mount_point)),
      ior_entries_(ior_entries),
      io_depth_(io_depth),
      numa_id_(numa_id),
      iov_size_(iov_size),
      time_out_(time_out),
      read_buffer_(read_buffer),
      read_buffer_size_(read_buffer_size),
      read_buffer_id_(std::move(read_buffer_id)),
      buffer_enabled_(enable_key_buffer) {
  if ((read_buffer_ == 0 && (read_buffer_size_ || !read_buffer_id_.empty())) ||
      (read_buffer_ != 0 &&
       (read_buffer_size_ == 0 || read_buffer_id_.size() != 16 ||
        read_buffer_size_ > UINTPTR_MAX - read_buffer_))) {
    throw std::invalid_argument("Invalid external HF3FS read arena");
  }
  if (immutable_fd_capacity > 65536)
    throw std::invalid_argument("FD cache limit exceeds 65536");
  // Debug print worker pool configuration
  fprintf(stderr,
          "[LMCache HF3FS] Worker pool: num_workers=%d, "
          "per_op_workers={",
          num_workers_);
  for (const auto& [key, count] : worker_pool_config_.per_op_workers) {
    fprintf(stderr, "%s=\"%d\"", key.c_str(), count);
  }
  fprintf(stderr, "}\n");

  // Validate mount_point using 3FS SDK
  char mount_point_buf[PATH_MAX];
  int ret = hf3fs_extract_mount_point(mount_point_buf, sizeof(mount_point_buf),
                                      mount_point_.c_str());
  if (ret < 0) {
    throw std::runtime_error("hf3fs_extract_mount_point failed: '" +
                             mount_point_ + "' is not a valid 3FS path");
  }
  if (ret > static_cast<int>(sizeof(mount_point_buf))) {
    throw std::runtime_error("Mount point path too long: '" + mount_point_ +
                             "'");
  }

  std::string extracted(mount_point_buf);
  if (extracted != mount_point_) {
    throw std::runtime_error("Mount point mismatch: config='" + mount_point_ +
                             "', extracted='" + extracted + "'");
  }

  // Parse comma-separated base_paths
  std::stringstream ss(base_paths);
  std::string path;
  while (std::getline(ss, path, ',')) {
    // Trim whitespace
    path.erase(0, path.find_first_not_of(" \t"));
    path.erase(path.find_last_not_of(" \t") + 1);
    if (!path.empty()) {
      base_paths_.push_back(path);
    }
  }

  if (base_paths_.empty()) {
    throw std::runtime_error("No valid base_paths provided");
  }

  // Validate each base_path is a subdirectory of mount_point
  for (const auto& bp : base_paths_) {
    if (bp.find(mount_point_) != 0) {
      throw std::runtime_error("base_path '" + bp +
                               "' is not a subdirectory of mount_point '" +
                               mount_point_ + "'");
    }
  }

  if (buffer_enabled_) {
    fprintf(stderr, "[LMCache HF3FS] Buffer enabled, begin scan...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    scan_and_build_buffer_();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr,
            "[LMCache HF3FS] End of scan, scanned %zu keys from %zu "
            "base_path(s), elapsed %ld ms\n",
            key_buffer_.size(), base_paths_.size(), elapsed_ms);
  } else {
    fprintf(stderr, "[LMCache HF3FS] Buffer disabled\n");
  }

  if (hash_shards_) {
    if (buffer_enabled_)
      throw std::invalid_argument("Hash shards require filesystem lookup");
    const char* hex = "0123456789abcdef";
    for (const auto& base : base_paths_) {
      std::filesystem::create_directories(base);
      directories_.add(base);
      for (unsigned i = 0; i < 256; ++i) {
        std::string bucket{hex[i >> 4], hex[i & 15]};
        std::string directory = base + "/.metadata-shards-v1/" + bucket;
        std::filesystem::create_directories(directory);
        directories_.add(directory);
      }
    }
  }
  // IMPORTANT: start worker threads at the END of the constructor
  start_workers();
}

/**
 * Initialize read Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_read_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             true,  // for_read=true
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_write_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             false,  // for_read=false
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize read Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_read_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_write_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Create a new per-thread connection.
 *
 * Initializes:
 * - Read Ior and Iov
 * - Write Ior and Iov
 * - File descriptors (initialized to -1)
 *
 * @return WorkerHf3fsConn Initialized connection structure
 */
WorkerHf3fsConn Hf3fsConnector::create_connection() {
  WorkerHf3fsConn conn;

  // Initialize read Ior
  init_read_ior(conn.read_ior);
  conn.read_ior_initialized = true;

  // Initialize write Ior
  init_write_ior(conn.write_ior);
  conn.write_ior_initialized = true;

  // Initialize read Iov
  if (read_buffer_) {
    const int ret =
        hf3fs_iovwrap(&conn.read_iov, reinterpret_cast<void*>(read_buffer_),
                      reinterpret_cast<const uint8_t*>(read_buffer_id_.data()),
                      mount_point_.c_str(), read_buffer_size_, 0, -1);
    if (ret < 0) throw std::runtime_error("Cannot wrap external HF3FS arena");
  } else {
    init_read_iov(conn.read_iov);
    conn.read_iov_initialized = true;
  }

  // Initialize write Iov
  init_write_iov(conn.write_iov);
  conn.write_iov_initialized = true;

  // File descriptors start uninitialized
  conn.fd = -1;

  return conn;
}

/**
 * Open a file for read or write.
 *
 * Steps:
 * 1. Open file with appropriate flags (O_RDONLY or O_WRONLY|O_CREAT)
 * 2. Register FD with 3FS via hf3fs_reg_fd
 * 3. Store file path in connection
 *
 * @param conn Connection to use
 * @param file_path Path to file
 * @param for_write true for write, false for read
 * @throws std::runtime_error if open or hf3fs_reg_fd fails
 */
void Hf3fsConnector::open_key(WorkerHf3fsConn& conn, const std::string& key,
                              bool write) {
  conn.object_offset = 0;
  if (!slot_key(key)) {
    open_file(conn, key_to_path(key), write);
    return;
  }
  if (key.size() != 31 || key[22] != '@')
    throw std::invalid_argument("Invalid slot key");
  for (size_t i = 6; i < key.size(); ++i)
    if (i != 22 &&
        !((key[i] >= '0' && key[i] <= '9') || (key[i] >= 'a' && key[i] <= 'f')))
      throw std::invalid_argument("Invalid slot coordinates");
  uint64_t shard = std::stoull(key.substr(6, 16), nullptr, 16);
  uint64_t slot = std::stoull(key.substr(23, 8), nullptr, 16);
  if (shard >= 1024 || slot >= 16384)
    throw std::invalid_argument("Slot outside namespace");
  auto path = base_paths_[0] + "/.slots-v1/" + key.substr(6, 16) + ".bin";
  conn.read_lease =
      read_files_.acquire(path, "", AT_FDCWD, "", AT_FDCWD, O_RDWR);
  conn.fd = conn.read_lease->fd;
  conn.registered = false;
  conn.file_path = path;
  conn.object_offset = slot * 262144ULL;
}

void Hf3fsConnector::open_file(WorkerHf3fsConn& conn,
                               const std::string& file_path, bool for_write) {
  conn.object_offset = 0;
  if (!for_write && read_files_.enabled()) {
    conn.read_lease = read_files_.acquire(
        file_path, legacy_path(file_path),
        hash_shards_ ? directory_fd(file_path) : AT_FDCWD,
        hash_shards_ ? std::filesystem::path(file_path).filename().string()
                     : "",
        hash_shards_ ? directory_fd(legacy_path(file_path)) : AT_FDCWD);
    conn.fd = conn.read_lease->fd;
    conn.registered = false;
    conn.file_path = file_path;
    return;
  }
  auto opened_at = metadata_now();
  std::string opened_path = file_path;
  if (for_write) {
    std::random_device random;
    do {
      // O_EXCL prevents collisions across workers, processes, and hosts.
      opened_path = file_path + "." + std::to_string(random());
      conn.fd =
          hash_shards_
              ? ::openat(directory_fd(file_path),
                         std::filesystem::path(opened_path).filename().c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644)
              : ::open(opened_path.c_str(),
                       O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    } while (conn.fd < 0 && errno == EEXIST);
  } else {
    conn.fd =
        hash_shards_
            ? ::openat(directory_fd(file_path),
                       std::filesystem::path(file_path).filename().c_str(),
                       O_RDONLY | O_CLOEXEC)
            : ::open(opened_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (conn.fd < 0 && errno == ENOENT && hash_shards_) {
      ++metadata_metrics_->opens;
      conn.fd = ::openat(directory_fd(legacy_path(file_path)),
                         std::filesystem::path(file_path).filename().c_str(),
                         O_RDONLY | O_CLOEXEC);
    }
  }
  metadata_metrics_->open_ns += metadata_now() - opened_at;
  ++metadata_metrics_->opens;
  if (conn.fd < 0) {
    throw std::runtime_error("open failed: " + std::string(strerror(errno)));
  }

  auto registered_at = metadata_now();
  int reg_result = hf3fs_reg_fd(conn.fd, 0);
  metadata_metrics_->reg_ns += metadata_now() - registered_at;
  ++metadata_metrics_->registrations;
  if (reg_result > 0) {
    ::close(conn.fd);
    conn.fd = -1;
    if (for_write && ::unlink(opened_path.c_str()) < 0) {
      fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
              opened_path.c_str(), strerror(errno));
    }
    throw std::runtime_error(
        "hf3fs_reg_fd failed: " + std::to_string(reg_result) +
        " (errno=" + strerror(reg_result) + ")");
  }
  conn.registered = true;
  conn.file_path = std::move(opened_path);
}

/**
 * Close a file and deregister from 3FS.
 *
 * Steps:
 * 1. Deregister FD from 3FS via hf3fs_dereg_fd
 * 2. Close Linux FD
 * 3. Clear file path
 *
 * @param conn Connection to use
 */
void Hf3fsConnector::close_file(WorkerHf3fsConn& conn) {
  if (conn.read_lease) {
    conn.fd = -1;
    conn.registered = false;
    conn.read_lease.reset();
    conn.file_path.clear();
    return;
  }
  // Step 1: Deregister FD from 3FS first (only if registration succeeded)
  if (conn.registered) {
    auto ts = metadata_now();
    hf3fs_dereg_fd(conn.fd);
    metadata_metrics_->dereg_ns += metadata_now() - ts;
    ++metadata_metrics_->deregistrations;
    conn.registered = false;
  }
  // Step 2: Close Linux FD
  if (conn.fd >= 0) {
    auto ts = metadata_now();
    ::close(conn.fd);
    metadata_metrics_->close_ns += metadata_now() - ts;
    conn.fd = -1;
  }
  conn.file_path.clear();
}

/**
 * Read data from a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Prepare I/O request via hf3fs_prep_io
 * 2. Submit I/O via hf3fs_submit_ios
 * 3. Wait for completion via hf3fs_wait_for_ios
 * 4. Copy data from Iov buffer to output buffer
 *
 * If len > iov_size_, data is read in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::read_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                               hf3fs_iov& iov, void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_read = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    int ret = hf3fs_prep_io(&ior, &iov, true, iov.base, conn.fd,
                            conn.object_offset + file_offset, sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    const int submitted = hf3fs_submit_ios(&ior);
    hf3fs_cqe cqe{};
    do {
      ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    } while (ret == 0 || ret == -EINTR);
    if (ret < 0) std::terminate();  // Cannot cancel already accepted DMA.
    if (submitted < 0)
      throw std::runtime_error("Submit failed after IO drained");
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }
    if (cqe.result != sub_len) {
      throw std::runtime_error(
          "read_file (" + conn.file_path + ") failed, requested " +
          std::to_string(sub_len) + " but got " + std::to_string(cqe.result));
    }
    memcpy(static_cast<char*>(buf) + total_read, iov.base, sub_len);
    total_read += sub_len;
    file_offset += sub_len;
  }
}

/**
 * Write data to a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Copy data to Iov buffer
 * 2. Prepare I/O request via hf3fs_prep_io
 * 3. Submit I/O via hf3fs_submit_ios
 * 4. Wait for completion via hf3fs_wait_for_ios
 *
 * If len > iov_size_, data is written in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::write_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                                hf3fs_iov& iov, const void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_written = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    // Copy data to Iov buffer
    memcpy(iov.base, static_cast<const char*>(buf) + file_offset, sub_len);

    int ret = hf3fs_prep_io(&ior, &iov, false, iov.base, conn.fd,
                            conn.object_offset + file_offset, sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    const int submitted = hf3fs_submit_ios(&ior);
    hf3fs_cqe cqe{};
    do {
      ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    } while (ret == 0 || ret == -EINTR);
    if (ret < 0) std::terminate();  // Cannot cancel already accepted DMA.
    if (submitted < 0)
      throw std::runtime_error("Submit failed after IO drained");
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }

    if (cqe.result != sub_len) {
      throw std::runtime_error(
          "write_file (" + conn.file_path + ") failed, requested " +
          std::to_string(sub_len) + " but got " + std::to_string(cqe.result));
    }
    total_written += sub_len;
    file_offset += sub_len;
  }
}

/**
 * Select a base path based on the key's ``chunk_hash``.
 *
 * Wire format:
 *   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>]
 *
 * Algorithm:
 * 1. Extract the ``chunk_hash_hex`` field via
 * ``keys.h::get_chunk_hash_from_key``.
 * 2. Use the LAST 16 hex chars (64 bits) of that hash: sequential keys
 *    like ``...00000000``, ``...00000001`` share leading zeros but differ
 *    in their low bits, so the tail carries the entropy.
 * 3. Convert to uint64 and mod by number of paths.
 *
 * @param key Key string
 * @return Selected base path
 */
const std::string& Hf3fsConnector::select_base_path(
    const std::string& key) const {
  if (base_paths_.size() == 1) {
    return base_paths_[0];
  }

  std::string hash_str;
  try {
    hash_str = get_chunk_hash_from_key(key);
  } catch (const std::exception& e) {
    fprintf(stderr, "[LMCache HF3FS] get chunk hash failed: %s\n", e.what());
    return base_paths_[0];
  }

  // Use last 16 characters (64 bits) for better distribution.
  size_t hash_len = hash_str.length();
  size_t substr_start = (hash_len > 16) ? (hash_len - 16) : 0;
  std::string hash_substr = hash_str.substr(substr_start);

  uint64_t hash_val = std::stoull(hash_substr, nullptr, 16);
  size_t idx = hash_val % base_paths_.size();

  return base_paths_[idx];
}

/**
 * Convert a key to a full file path.
 *
 * The filename is produced by  ``keys.h::key_to_filename`` The base path is
 * selected by hashing the key's ``chunk_hash`` across ``base_paths_``.
 *
 * @param key Key string (wire format ``model@kv_rank@ogid@chunk_hash[@salt]``)
 * @return Full file path (base_path + filename)
 */
std::string Hf3fsConnector::key_to_path(const std::string& key) {
  if (slot_key(key)) return base_paths_[0] + "/.slots-v1/internal";
  const std::string& base_path = select_base_path(key);
  std::string filename = key_to_filename(key);
  if (hash_shards_) {
    const auto hash = get_chunk_hash_from_key(key);
    if (hash.size() < 2)
      throw std::invalid_argument("Short chunk hash for sharding");
    std::string bucket = hash.substr(hash.size() - 2);
    for (auto& c : bucket) {
      if (c >= 'A' && c <= 'F') c += ('a' - 'A');
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
        throw std::invalid_argument("Non-hex chunk hash for sharding");
    }
    return base_path + "/.metadata-shards-v1/" + bucket + "/" + filename;
  }
  return base_path + "/" + filename;
}

/**
 * Retrieve data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open file for reading
 * 3. Read data using 3FS I/O
 * 4. Close file
 *
 * @param conn Connection to use
 * @param key Key to retrieve
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_get(WorkerHf3fsConn& conn,
                                   const std::string& key, void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS
  std::string file_path = key_to_path(key);
  if (slot_key(key) && len != 229376)
    throw std::invalid_argument("Slot pack size");
  open_key(conn, key, false);
  try {
    read_file(conn, conn.read_ior, conn.read_iov, buf, len);
  } catch (const std::exception& e) {
    read_files_.invalidate(file_path);
    close_file(conn);
    throw;
  }
  close_file(conn);
}

void Hf3fsConnector::do_batch_get(WorkerHf3fsConn& conn, const Request& req) {
  struct ReadSlot {
    WorkerHf3fsConn file;
    size_t item = 0;
    size_t offset = 0;
  };

  // Leave one ring entry free: the shared submission/completion queues use
  // an empty slot to distinguish full from empty. A worker owns its ring.
  const int entries = hf3fs_io_entries(&conn.read_ior);
  if (entries < 2 || conn.read_iov.size == 0) {
    throw std::runtime_error("Invalid HF3FS read ring or buffer capacity");
  }
  const size_t limit = std::min(req.keys.size(), size_t(entries - 1));
  // Allocate before preparing any I/O. Slot addresses are stable userdata,
  // and no C++ allocation/exception may unwind a window with outstanding I/O.
  std::vector<ReadSlot> slots(limit);
  std::vector<hf3fs_cqe> completions(limit);
  size_t next = 0;
  while (next < req.keys.size()) {
    const size_t len = req.buf_lens[next];
    if (read_buffer_ && (len == 0 || len > read_buffer_size_)) {
      ++next;
      continue;
    }
    if (len == 0 || len > conn.read_iov.size) {
      // Preserve the existing segmented read for objects larger than the Iov.
      // The preceding window is fully drained before this buffer is reused.
      try {
        do_single_get(conn, req.keys[next], req.buf_ptrs[next], len,
                      req.batch_chunk_num_bytes);
        req.batch->per_key_results[req.start_idx + next] = 1;
      } catch (const std::exception& e) {
        fprintf(stderr, "[LMCache HF3FS GET] key %s failed: %s\n",
                req.keys[next].c_str(), e.what());
      }
      ++next;
      continue;
    }

    size_t count = 0;
    size_t used = 0;
    while (next < req.keys.size() && count < limit) {
      const size_t size = req.buf_lens[next];
      if (read_buffer_) {
        const auto target = reinterpret_cast<uintptr_t>(req.buf_ptrs[next]);
        if (size == 0 || target < read_buffer_ ||
            target - read_buffer_ > read_buffer_size_ ||
            size > read_buffer_size_ - (target - read_buffer_)) {
          ++next;
          continue;
        }
      } else if (size == 0 || size > conn.read_iov.size - used) {
        break;
      }
      auto& slot = slots[count++];
      slot.item = next++;
      slot.offset = read_buffer_
                        ? reinterpret_cast<uintptr_t>(req.buf_ptrs[slot.item]) -
                              read_buffer_
                        : used;
      if (!read_buffer_) used += size;
      // Finish potentially throwing path/FD setup before preparing the window.
      try {
        if (slot_key(req.keys[slot.item]) && size != 229376)
          throw std::invalid_argument("Slot pack size");
        open_key(slot.file, req.keys[slot.item], false);
      } catch (const std::exception& e) {
        close_file(slot.file);
        fprintf(stderr, "[LMCache HF3FS GET] key %s failed: %s\n",
                req.keys[slot.item].c_str(), e.what());
      }
    }

    size_t pending = 0;
    for (size_t i = 0; i < count; ++i) {
      auto& slot = slots[i];
      if (slot.file.fd < 0) continue;
      const int ret = hf3fs_prep_io(&conn.read_ior, &conn.read_iov, true,
                                    conn.read_iov.base + slot.offset,
                                    slot.file.fd, slot.file.object_offset,
                                    req.buf_lens[slot.item], &slot);
      if (ret < 0) {
        fprintf(stderr, "[LMCache HF3FS GET] prep failed for %s: %d\n",
                req.keys[slot.item].c_str(), -ret);
        continue;
      }
      ++pending;
    }

    if (pending) {
      // Usrbio can process prepared entries before this notification. Even if
      // the notification fails, every prepared entry must be reaped before
      // closing FDs or reusing buffers. Report such a window as failed.
      const int submitted = hf3fs_submit_ios(&conn.read_ior);
      if (submitted < 0) {
        fprintf(stderr, "[LMCache HF3FS GET] submit hint failed: %d\n",
                -submitted);
      }
      while (pending) {
        const int done =
            hf3fs_wait_for_ios(&conn.read_ior, completions.data(),
                               static_cast<int>(pending), 1, nullptr);
        if (done == -EINTR) continue;
        if (done < 0) {
          // The SDK only returns EINVAL for an invalid ring/argument here.
          // There is no cancellation API: returning a batch completion would
          // let callers reclaim buffers still used by I/O. Fail closed.
          fprintf(stderr, "[LMCache HF3FS GET] cannot drain read ring: %d\n",
                  -done);
          std::terminate();
        }
        for (int i = 0; i < done; ++i) {
          const auto& cqe = completions[i];
          const auto& slot = *static_cast<const ReadSlot*>(cqe.userdata);
          // Each accepted prep has exactly one completion and returns its
          // userdata; the order need not match submission order.
          const size_t size = req.buf_lens[slot.item];
          if (submitted >= 0 && cqe.result == static_cast<int64_t>(size)) {
            if (!read_buffer_) {
              memcpy(req.buf_ptrs[slot.item], conn.read_iov.base + slot.offset,
                     size);
            }
            req.batch->per_key_results[req.start_idx + slot.item] = 1;
          } else {
            read_files_.invalidate(slot.file.file_path);
            fprintf(stderr,
                    "[LMCache HF3FS GET] key %s read %ld, expected %zu bytes\n",
                    req.keys[slot.item].c_str(), cqe.result, size);
          }
          --pending;
        }
      }
    }
    for (size_t i = 0; i < count; ++i) close_file(slots[i].file);
  }
}

/**
 * Store data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open a unique temporary file for writing
 * 3. Write data using 3FS I/O
 * 4. Close file and atomically rename to final path
 * 5. Update key buffer
 *
 * @param conn Connection to use
 * @param key Key to store
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_set(WorkerHf3fsConn& conn,
                                   const std::string& key, const void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS

  std::string file_path = key_to_path(key);
  if (slot_key(key)) {
    if (len != 229376) throw std::invalid_argument("Slot pack size");
    open_key(conn, key, true);
    try {
      write_file(conn, conn.write_ior, conn.write_iov, buf, len);
    } catch (...) {
      close_file(conn);
      throw;
    }
    close_file(conn);
    return;  // Caller publishes FDB index only after CQE.
  }
  read_files_.invalidate(file_path);
  open_file(conn, file_path, true);
  const std::string temporary_path = conn.file_path;
  try {
    write_file(conn, conn.write_ior, conn.write_iov, buf, len);
    close_file(conn);
    auto ts = metadata_now();
    if (hash_shards_) {
      int dir = directory_fd(file_path);
      if (::renameat(
              dir, std::filesystem::path(temporary_path).filename().c_str(),
              dir, std::filesystem::path(file_path).filename().c_str()) < 0)
        throw std::runtime_error("Atomic shard publish failed: " +
                                 std::string(strerror(errno)));
    } else {
      std::filesystem::rename(temporary_path, file_path);
    }
    metadata_metrics_->rename_ns += metadata_now() - ts;
    ++metadata_metrics_->renames;
    read_files_.invalidate(file_path);
  } catch (const std::exception& e) {
    close_file(conn);
    if (::unlink(temporary_path.c_str()) < 0) {
      fprintf(stderr, "[LMCache HF3FS] Remove temporary file %s failed: %s\n",
              temporary_path.c_str(), strerror(errno));
    }
    throw;
  }
  if (buffer_enabled_) {
    buffer_add_(key);
  }
}

/**
 * Check if a key exists.
 *
 * @param conn Connection to use
 * @param key Key to check
 * @return true if key exists, false otherwise
 */
bool Hf3fsConnector::do_single_exists(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  if (!buffer_enabled_) {
    std::string file_path = key_to_path(key);
    return std::filesystem::exists(file_path) ||
           (hash_shards_ && std::filesystem::exists(legacy_path(file_path)));
  }
  return buffer_contains_(key);
}

/**
 * Delete a key.
 *
 * @param conn Connection to use
 * @param key Key to delete
 * @return true if deleted, false if not found
 */
bool Hf3fsConnector::do_single_delete(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  std::string file_path = key_to_path(key);
  try {
    read_files_.invalidate(file_path);
    bool removed = std::filesystem::remove(file_path);
    if (hash_shards_)
      removed = std::filesystem::remove(legacy_path(file_path)) || removed;
    read_files_.invalidate(file_path);
    if (removed && buffer_enabled_) {
      buffer_remove_(key);
    }
    return removed;
  } catch (const std::filesystem::filesystem_error& e) {
    fprintf(stderr, "[LMCache HF3FS] Delete file %s failed: %s\n",
            file_path.c_str(), e.what());
    return false;
  }
}

/**
 * Scan all base_paths in parallel and populate the key buffer.
 *
 * Splits base_paths_ into num_workers_ contiguous slices (bounded by the
 * number of base_paths). Each thread scans one slice of directories into a
 * private local_set to avoid contention, then all local_sets are merged into
 * key_buffer_ in a single-threaded pass.
 *
 * Called from the constructor (single-threaded, before workers start).
 */
void Hf3fsConnector::scan_and_build_buffer_() {
  if (base_paths_.empty()) {
    return;
  }

  int num_workers =
      std::min(num_workers_, static_cast<int>(base_paths_.size()));
  if (num_workers <= 0) {
    num_workers = 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_workers));
  std::vector<std::unordered_set<std::string>> local_sets(num_workers);

  for (int i = 0; i < num_workers; ++i) {
    // slice sizes differ by at most one.
    size_t start = static_cast<size_t>(i) * base_paths_.size() /
                   static_cast<size_t>(num_workers);
    size_t end = static_cast<size_t>(i + 1) * base_paths_.size() /
                 static_cast<size_t>(num_workers);

    threads.emplace_back(
        &Hf3fsConnector::buffer_scan_worker_, this,
        std::vector<std::string>(
            base_paths_.begin() + static_cast<std::ptrdiff_t>(start),
            base_paths_.begin() + static_cast<std::ptrdiff_t>(end)),
        std::ref(local_sets[i]));
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Single-threaded merge -- no contention
  for (int i = 0; i < num_workers; ++i) {
    for (const auto& key : local_sets[i]) {
      key_buffer_.Insert(key);
    }
  }
}

/**
 * Scan the given base_path directories and recover keys from .data filenames.
 *
 * This keeps directory traversal local to the connector; the per-file mapping
 * from filename to the original wire key is delegated to the shared
 * ``keys.h::filename_to_key`` (the inverse of ``keys.h::key_to_filename``),
 * so the on-disk encoding stays consistent with the fs_native connector.
 *
 * Each path in base_paths is iterated in turn; within one directory, every
 * regular ".data" file that decodes to a well-formed key is inserted into the
 * provided local set.
 *
 * Called from worker threads spawned in scan_and_build_buffer_().
 *
 * @param base_paths Directories to scan for .data files
 * @param local_set Set to populate with recovered keys
 */
void Hf3fsConnector::buffer_scan_worker_(
    const std::vector<std::string>& base_paths,
    std::unordered_set<std::string>& local_set) {
  std::error_code ec;
  for (const auto& base_path : base_paths) {
    for (const auto& entry :
         std::filesystem::directory_iterator(base_path, ec)) {
      if (!entry.is_regular_file(ec)) {
        continue;
      }

      const std::string filename = entry.path().filename().string();
      std::string recovered = filename_to_key(filename);
      if (!recovered.empty()) {
        local_set.insert(std::move(recovered));
      }
    }
  }
}

/**
 * Add a key to the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_set().
 */
void Hf3fsConnector::buffer_add_(const std::string& key) {
  key_buffer_.Insert(key);
}

/**
 * Remove a key from the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_delete().
 */
void Hf3fsConnector::buffer_remove_(const std::string& key) {
  key_buffer_.Erase(key);
}

/**
 * Check if a key exists in the buffer.
 *
 * Thread-safe read using sharded_flat_hash_set::Contains().
 * Each shard has its own lock, reducing contention.
 *
 * @param key Key to check
 * @return true if key is in the buffer, false otherwise
 */
bool Hf3fsConnector::buffer_contains_(const std::string& key) const {
  return key_buffer_.Contains(key);
}

/**
 * Optimized batch exists using the in-memory key buffer.
 *
 * When the buffer is enabled, all lookups are answered from the hash
 * set in microseconds. When disabled, falls back to the base class
 * default which iterates do_single_exists (filesystem).
 */
void Hf3fsConnector::do_batch_exists(WorkerHf3fsConn& conn,
                                     const Request& req) {
  (void)conn;
  if (!buffer_enabled_) {
    ConnectorBase::do_batch_exists(conn, req);
    return;
  }
  // Buffer path: check cache for all keys
  for (size_t i = 0; i < req.keys.size(); ++i) {
    req.batch->per_key_results[req.start_idx + i] =
        buffer_contains_(req.keys[i]) ? 1 : 0;
  }
}

/**
 * Shutdown all connections.
 *
 * Worker destructors handle cleanup automatically,
 * so this method is a no-op.
 */
void Hf3fsConnector::shutdown_connections() {
  // Worker destructors handle cleanup
}

}  // namespace connector
}  // namespace lmcache
