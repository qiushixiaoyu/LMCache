// SPDX-License-Identifier: Apache-2.0
// Opt-in leases for immutable, no-eviction canonical objects. No payload cache.
#pragma once
#include <fcntl.h>
#include <hf3fs_usrbio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lmcache::connector {
inline uint64_t metadata_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
struct MetadataMetrics {
  std::atomic<uint64_t> open_ns{0}, reg_ns{0}, dereg_ns{0}, close_ns{0},
      rename_ns{0};
  std::atomic<uint64_t> opens{0}, registrations{0}, deregistrations{0},
      renames{0};
  std::atomic<uint64_t> hits{0}, misses{0}, evictions{0}, invalidations{0},
      live{0};
  std::map<std::string, uint64_t> snapshot() const {
    return {{"open_ns", open_ns.load()},
            {"reg_ns", reg_ns.load()},
            {"dereg_ns", dereg_ns.load()},
            {"close_ns", close_ns.load()},
            {"rename_ns", rename_ns.load()},
            {"opens", opens.load()},
            {"registrations", registrations.load()},
            {"deregistrations", deregistrations.load()},
            {"renames", renames.load()},
            {"fd_hits", hits.load()},
            {"fd_misses", misses.load()},
            {"fd_evictions", evictions.load()},
            {"fd_invalidations", invalidations.load()},
            {"fd_live", live.load()}};
  }
};
struct RegisteredFile {
  int fd;
  std::shared_ptr<MetadataMetrics> metrics;
  RegisteredFile(int value, std::shared_ptr<MetadataMetrics> m)
      : fd(value), metrics(std::move(m)) {
    ++metrics->live;
  }
  RegisteredFile(const RegisteredFile&) = delete;
  ~RegisteredFile() {
    auto t = metadata_now();
    hf3fs_dereg_fd(fd);
    metrics->dereg_ns += metadata_now() - t;
    ++metrics->deregistrations;
    t = metadata_now();
    ::close(fd);
    metrics->close_ns += metadata_now() - t;
    --metrics->live;
  }
};

class RegisteredFiles {
  using Lease = std::shared_ptr<RegisteredFile>;
  struct Entry {
    Lease file;
    std::list<std::string>::iterator pos;
  };
  const size_t capacity_;
  std::shared_ptr<MetadataMetrics> metrics_;
  mutable std::mutex mu_;
  uint64_t epoch_ = 0;
  std::list<std::string> order_;
  std::unordered_map<std::string, Entry> files_;

 public:
  RegisteredFiles(size_t capacity, std::shared_ptr<MetadataMetrics> m)
      : capacity_(capacity), metrics_(std::move(m)) {}
  bool enabled() const { return capacity_ != 0; }
  Lease acquire(const std::string& path, const std::string& fallback = "",
                int directory = AT_FDCWD, const std::string& leaf = "",
                int fallback_directory = AT_FDCWD, int flags = O_RDONLY) {
    uint64_t epoch;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto found = files_.find(path);
      if (found != files_.end()) {
        order_.splice(order_.end(), order_, found->second.pos);
        ++metrics_->hits;
        return found->second.file;
      }
      epoch = epoch_;
      ++metrics_->misses;
    }
    // Slow FUSE operations run without the LRU mutex. Concurrent cold misses
    // may open the same file; only one resulting lease is retained in the LRU.
    auto t = metadata_now();
    int fd = ::openat(directory, leaf.empty() ? path.c_str() : leaf.c_str(),
                      flags | O_CLOEXEC);
    const int open_error = errno;
    metrics_->open_ns += metadata_now() - t;
    ++metrics_->opens;
    if (fd < 0 && open_error == ENOENT && !fallback.empty()) {
      t = metadata_now();
      fd = ::openat(
          fallback_directory,
          fallback_directory == AT_FDCWD ? fallback.c_str() : leaf.c_str(),
          O_RDONLY | O_CLOEXEC);
      metrics_->open_ns += metadata_now() - t;
      ++metrics_->opens;
    }
    if (fd < 0) throw std::runtime_error("cached open failed: " + path);
    if (flags == O_RDWR) {
      struct stat st{};
      if (fstat(fd, &st) != 0 || st.st_size != 4294967296LL) {
        ::close(fd);
        throw std::runtime_error("Invalid slot file size: " + path);
      }
    }
    t = metadata_now();
    int rc = hf3fs_reg_fd(fd, 0);
    metrics_->reg_ns += metadata_now() - t;
    ++metrics_->registrations;
    if (rc > 0) {
      ::close(fd);
      throw std::runtime_error("cached FD registration failed: " +
                               std::to_string(rc));
    }
    Lease opened;
    try {
      opened = std::make_shared<RegisteredFile>(fd, metrics_);
    } catch (...) {
      hf3fs_dereg_fd(fd);
      ::close(fd);
      throw;
    }
    Lease answer = opened, retired;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (epoch != epoch_)
        return opened;  // A mutation raced this open: do not retain it.
      auto found = files_.find(path);
      if (found != files_.end()) return found->second.file;
      order_.push_back(path);
      try {
        files_.emplace(path, Entry{opened, std::prev(order_.end())});
      } catch (...) {
        order_.pop_back();
        throw;
      }
      if (files_.size() > capacity_) {
        auto old = files_.find(order_.front());
        retired = std::move(old->second.file);
        files_.erase(old);
        order_.pop_front();
        ++metrics_->evictions;
      }
    }
    // Last lease closes outside the lock and only after its I/O completion.
    return answer;
  }
  void invalidate(const std::string& path) {
    Lease retired;
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++epoch_;
      ++metrics_->invalidations;
      auto found = files_.find(path);
      if (found != files_.end()) {
        retired = std::move(found->second.file);
        order_.erase(found->second.pos);
        files_.erase(found);
      }
    }
  }
  void clear() {
    decltype(files_) retired;
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++epoch_;
      retired.swap(files_);
      order_.clear();
    }
  }
  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return files_.size();
  }
};

// All directory handles are acquired before workers start, then read-only.
// Moving/rebuilding the namespace requires draining and restarting clients.
class DirectoryHandles {
  std::unordered_map<std::string, int> handles_;

 public:
  ~DirectoryHandles() {
    for (const auto& item : handles_) ::close(item.second);
  }
  void add(const std::string& path) {
    int fd = ::open(path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("Cannot pin shard directory: " + path);
    try {
      auto result = handles_.emplace(path, fd);
      if (!result.second) ::close(fd);
    } catch (...) {
      ::close(fd);
      throw;
    }
  }
  int get(const std::string& path) const { return handles_.at(path); }
};
}  // namespace lmcache::connector
