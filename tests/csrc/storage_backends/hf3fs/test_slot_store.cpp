// SPDX-License-Identifier: Apache-2.0
// Exercise the public connector API against a deterministic Usrbio test double.
// Compile with connector.cpp and the SDK header, without linking the SDK
// library.
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "connector.h"

namespace {
struct Read {
  bool is_read;
  int fd;
  void* buffer;
  size_t offset;
  size_t size;
  const void* userdata;
};
struct Ring {
  int entries;
  std::vector<Read> reads;
};
std::mutex mock_mutex;
std::set<int> registered;
std::vector<Ring*> rings;
size_t prep_calls = 0, submit_calls = 0, peak = 0;
size_t fail_prep_at = 0, fail_register_at = 0, register_calls = 0;
bool interrupt_wait = false, fatal_wait = false, fail_submit = false;
bool completion_error = false;

void reset() {
  assert(registered.empty() && rings.empty());
  prep_calls = submit_calls = peak = register_calls = 0;
  fail_prep_at = fail_register_at = 0;
  interrupt_wait = fatal_wait = fail_submit = completion_error = false;
}

std::string wire_key(size_t i) { return "batch@1@0@" + std::to_string(i); }
std::string filename(const std::string& path, size_t i) {
  return path + "/batch@0x1@0@" + std::to_string(i) + ".data";
}

struct Fixture {
  std::string path;
  std::vector<std::string> keys;
  std::vector<std::vector<char>> data, output;
  std::vector<void*> buffers;
  std::vector<size_t> sizes;

  explicit Fixture(const std::vector<size_t>& lengths) : sizes(lengths) {
    char temp[] = "/tmp/lmcache-hf3fs-batch-XXXXXX";
    path = mkdtemp(temp);
    for (size_t i = 0; i < sizes.size(); ++i) {
      keys.push_back(wire_key(i));
      data.emplace_back(sizes[i]);
      for (size_t j = 0; j < sizes[i]; ++j) data.back()[j] = (i * 31 + j) % 127;
      std::ofstream file(filename(path, i), std::ios::binary);
      file.write(data.back().data(), data.back().size());
      output.emplace_back(sizes[i], char(0xa5));
    }
    for (auto& value : output) buffers.push_back(value.data());
  }
  ~Fixture() { std::filesystem::remove_all(path); }

  lmcache::connector::Completion read(lmcache::connector::Hf3fsConnector& c) {
    const auto id = c.submit_batch_get(keys, buffers, sizes, 0);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      auto results = c.drain_completions();
      if (!results.empty()) {
        assert(results.size() == 1 && results[0].future_id == id);
        return results[0];
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("Public GET completion timed out");
  }

  void check(const lmcache::connector::Completion& result,
             const std::set<size_t>& failed = {}) {
    assert(result.ok && result.result_bytes.size() == sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      assert(result.result_bytes[i] == !failed.count(i));
      if (!failed.count(i)) assert(output[i] == data[i]);
      if (failed.count(i))
        assert(std::all_of(output[i].begin(), output[i].end(),
                           [](char value) { return value == char(0xa5); }));
    }
  }
};
}  // namespace

extern "C" {
int hf3fs_extract_mount_point(char* out, int size, const char* path) {
  assert(strlen(path) + 1 <= size_t(size));
  strcpy(out, path);
  return strlen(path) + 1;
}
int hf3fs_iovcreate(hf3fs_iov* iov, const char*, size_t size, size_t, int) {
  iov->base = new uint8_t[size];
  iov->size = size;
  return 0;
}
int hf3fs_iovwrap(hf3fs_iov* iov, void* base, const uint8_t*, const char*,
                  size_t size, size_t, int) {
  iov->base = static_cast<uint8_t*>(base);
  iov->size = size;
  return 0;
}
void hf3fs_iovdestroy(hf3fs_iov* iov) { delete[] iov->base; }
int hf3fs_iorcreate4(hf3fs_ior* ior, const char*, int entries, bool for_read,
                     int, int, int, uint64_t) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  auto* ring = new Ring{entries, {}};
  ior->iorh = ring;
  ior->for_read = for_read;
  rings.push_back(ring);
  return 0;
}
void hf3fs_iordestroy(hf3fs_ior* ior) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  auto* ring = static_cast<Ring*>(ior->iorh);
  assert(ring->reads.empty());
  rings.erase(std::find(rings.begin(), rings.end(), ring));
  delete ring;
}
int hf3fs_io_entries(const hf3fs_ior* ior) {
  return static_cast<Ring*>(ior->iorh)->entries;
}
int hf3fs_reg_fd(int fd, uint64_t) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  if (++register_calls == fail_register_at) return EIO;
  assert(registered.insert(fd).second);
  return 0;
}
void hf3fs_dereg_fd(int fd) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  for (auto* ring : rings)
    for (const auto& read : ring->reads) assert(read.fd != fd);
  assert(registered.erase(fd) == 1);
}
int hf3fs_prep_io(const hf3fs_ior* ior, const hf3fs_iov* iov, bool read,
                  void* buffer, int fd, size_t offset, uint64_t size,
                  const void* userdata) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  assert(read == ior->for_read && registered.count(fd));
  auto* ptr = static_cast<uint8_t*>(buffer);
  assert(size && ptr >= iov->base && size <= iov->size - (ptr - iov->base));
  if (++prep_calls == fail_prep_at) return -EIO;
  auto* ring = static_cast<Ring*>(ior->iorh);
  if (ring->reads.size() == size_t(ring->entries - 1)) return -EAGAIN;
  for (const auto& pending : ring->reads) {
    const auto* begin = static_cast<uint8_t*>(pending.buffer);
    assert(ptr + size <= begin || begin + pending.size <= ptr);
  }
  ring->reads.push_back({read, fd, buffer, offset, size, userdata});
  peak = std::max(peak, ring->reads.size());
  return ring->reads.size() - 1;
}
int hf3fs_submit_ios(const hf3fs_ior*) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  ++submit_calls;
  if (fail_submit) {
    fail_submit = false;
    return -EIO;
  }
  return 0;
}
int hf3fs_wait_for_ios(const hf3fs_ior* ior, hf3fs_cqe* cqes, int capacity, int,
                       const timespec*) {
  std::lock_guard<std::mutex> lock(mock_mutex);
  if (fatal_wait) return -EINVAL;
  if (interrupt_wait) {
    interrupt_wait = false;
    return -EINTR;
  }
  auto* ring = static_cast<Ring*>(ior->iorh);
  // Reverse completion order and return partial batches, even when more are
  // ready.
  int count = std::min({capacity, 2, int(ring->reads.size())});
  for (int i = 0; i < count; ++i) {
    const auto read = ring->reads.back();
    ring->reads.pop_back();
    cqes[i].userdata = read.userdata;
    cqes[i].result = read.is_read
                         ? pread(read.fd, read.buffer, read.size, read.offset)
                         : pwrite(read.fd, read.buffer, read.size, read.offset);
    if (completion_error) {
      cqes[i].result = -EIO;
      completion_error = false;
    }
  }
  return count;
}
}  // extern "C"

int main() {
  using lmcache::connector::Hf3fsConnector;
  size_t tests = 0;
  auto pass = [&](const char* name) {
    assert(registered.empty() && rings.empty());
    std::cout << "PASS " << name << '\n';
    ++tests;
  };
  {
    reset();
    Fixture f({1, 3, 17, 9, 31, 1, 3, 17, 9, 31, 0, 80, 7});
    f.keys[5] = f.keys[0];
    f.data[5] = f.data[0];
    Hf3fsConnector c(f.path, f.path, 1, 8, 0, -1, 64, 200, false);
    f.check(f.read(c));
    c.close();
    assert(peak > 1 && peak <= 7);
  }
  pass(
      "mixed sizes, duplicate keys, reversed/partial completions, oversized "
      "and empty objects");
  {
    reset();
    Fixture f(std::vector<size_t>(50, 1));
    Hf3fsConnector c(f.path, f.path, 1, 8, 0, -1, 4096, 200, false);
    f.check(f.read(c));
    c.close();
    assert(peak == 7 && submit_calls == 8);
  }
  pass("ring limit and rollover");
  {
    reset();
    Fixture f(std::vector<size_t>(12, 9));
    Hf3fsConnector c(f.path, f.path, 1, 128, 0, -1, 32, 200, false);
    f.check(f.read(c));
    c.close();
    assert(peak == 3 && submit_calls == 4);
  }
  pass("byte budget and disjoint buffer slices");
  {
    reset();
    Fixture f(std::vector<size_t>(8, 8));
    std::filesystem::remove(filename(f.path, 1));
    std::filesystem::resize_file(filename(f.path, 4), 2);
    fail_register_at = 3;     // item 3; item 1 fails before registration
    fail_prep_at = 2;         // item 2
    completion_error = true;  // reversed first completion is item 7
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 1024, 200, false);
    f.check(f.read(c), {1, 2, 3, 4, 7});
    assert(registered.empty());
    c.close();
  }
  pass(
      "missing file, registration/prep failure, short read and negative "
      "completion");
  {
    reset();
    Fixture f(std::vector<size_t>(8, 8));
    fail_submit = true;
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 1024, 200, false);
    f.check(f.read(c), {0, 1, 2, 3, 4, 5, 6, 7});
    f.check(f.read(c));
    c.close();
  }
  pass("submit-hint failure drains all prepared I/O; next read succeeds");
  {
    reset();
    Fixture f(std::vector<size_t>(8, 8));
    interrupt_wait = true;
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 1024, 200, false);
    f.check(f.read(c));
    c.close();
  }
  pass("interrupted wait retries without releasing live buffers");
  {
    reset();
    Fixture f(std::vector<size_t>(60, 8));
    Hf3fsConnector c(f.path, f.path, 2, 16, 0, -1, 128, 200, false);
    f.check(f.read(c));
    c.close();
  }
  pass("multiple workers preserve caller order and per-key results");
  {
    reset();
    Fixture f(std::vector<size_t>(60, 8));
    {
      Hf3fsConnector c(f.path, f.path, 2, 16, 0, -1, 128, 200, false);
      c.submit_batch_get(f.keys, f.buffers, f.sizes, 0);
    }
    assert(f.output == f.data);
  }
  pass("destruction joins workers and drains outstanding requests");
  {
    reset();
    Fixture f(std::vector<size_t>(4, 8));
    const pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
      rlimit limit{0, 0};
      setrlimit(RLIMIT_CORE, &limit);
      fatal_wait = true;
      Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 1024, 200, false);
      f.read(c);
      _exit(1);
    }
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  }
  pass(
      "invalid completion API fails closed instead of reusing pending buffers");
  {
    reset();
    Fixture f(std::vector<size_t>(60, 8));
    std::vector<char> arena(1024, char(0xa5));
    for (size_t i = 0; i < 60; ++i) f.buffers[i] = arena.data() + i * 16;
    Hf3fsConnector c(f.path, f.path, 2, 16, 0, -1, 128, 200, false, {},
                     reinterpret_cast<uintptr_t>(arena.data()), arena.size(),
                     std::string(16, 'x'));
    auto result = f.read(c);
    for (size_t i = 0; i < 60; ++i) {
      assert(result.result_bytes[i]);
      assert(memcmp(arena.data() + i * 16, f.data[i].data(), 8) == 0);
      assert(arena[i * 16 + 8] == char(0xa5));
    }
    // Each invalid destination is rejected without preparing I/O for it.
    f.buffers[0] = f.output[0].data();
    f.buffers[1] = arena.data() + arena.size() - 4;
    f.buffers[2] = arena.data() + arena.size();
    result = f.read(c);
    for (size_t i = 0; i < 60; ++i) assert(result.result_bytes[i] == (i >= 3));
    assert(f.output[0] == std::vector<char>(8, char(0xa5)));
    c.close();
    arena[0] = 'z';  // Connector destruction must not free caller-owned memory.
  }
  pass(
      "external registered arena: multiworker direct targets, bounds, "
      "ownership");
  {
    reset();
    Fixture f(std::vector<size_t>(16, 8));
    Hf3fsConnector c(f.path, f.path, 2, 16, 0, -1, 128, 200, false, {}, 0, 0,
                     "", 32);
    f.check(f.read(c));
    auto before = c.metadata_stats();
    f.check(f.read(c));
    auto after = c.metadata_stats();
    assert(after["registrations"] == before["registrations"]);
    assert(after["fd_hits"] >= 16 && after["fd_cached"] == 16);
    c.clear_fd_cache();
    assert(c.metadata_stats()["fd_live"] == 0);
    f.check(f.read(c));
    c.close();
  }
  pass("registered FD reuse, explicit clear, exact bytes, clean close");
  {
    reset();
    Fixture f(std::vector<size_t>(60, 8));
    Hf3fsConnector c(f.path, f.path, 4, 16, 0, -1, 128, 200, false, {}, 0, 0,
                     "", 3);
    f.check(f.read(c));
    auto stats = c.metadata_stats();
    assert(stats["fd_cached"] <= 3 && stats["fd_live"] <= 3);
    assert(stats["fd_evictions"] > 0);
    f.check(f.read(c));
    c.close();
  }
  pass("LRU eviction never closes an outstanding read lease");
  {
    reset();
    Fixture f(std::vector<size_t>(8, 8));
    std::filesystem::remove(filename(f.path, 0));
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 128, 200, false, {}, 0, 0,
                     "", 16);
    f.check(f.read(c), {0});
    {
      std::ofstream out(filename(f.path, 0));
      out.write(f.data[0].data(), 8);
    }
    f.check(f.read(c));
    c.close();
  }
  pass("negative results are not cached; new publication becomes visible");
  {
    reset();
    Fixture f(std::vector<size_t>(8, 8));
    fail_register_at = 1;
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 128, 200, false, {}, 0, 0,
                     "", 16);
    f.check(f.read(c), {0});
    fail_register_at = 0;
    f.check(f.read(c));
    completion_error = true;
    f.read(c);
    completion_error = false;
    f.check(f.read(c));
    c.close();
  }
  pass("registration and I/O failures evict entries and preserve retry");
  for (int mode = 0; mode < 5; ++mode) {
    reset();
    Fixture f(std::vector<size_t>(3, 229376));
    std::filesystem::create_directory(f.path + "/.slots-v1");
    std::string segment = f.path + "/.slots-v1/0000000000000000.bin";
    int fd = ::open(segment.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    assert(fd >= 0 && ftruncate(fd, 4294967296LL) == 0);
    ::close(fd);
    f.keys = {"SLOT1@0000000000000000@00000002",
              "SLOT1@0000000000000000@00000004",
              "SLOT1@0000000000000000@00000006"};
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 1048576, 200, false, {}, 0,
                     0, "", 16);
    std::vector<void*> input;
    for (auto& data : f.data) input.push_back(data.data());
    if (mode == 1) fail_submit = true;
    if (mode == 2) interrupt_wait = true;
    if (mode == 3) completion_error = true;
    if (mode == 4) fail_prep_at = 1;
    auto id = c.submit_batch_set(f.keys, input, f.sizes, 0);
    lmcache::connector::Completion completed{};
    bool got = false;
    for (int i = 0; i < 5000 && !got; ++i) {
      auto results = c.drain_completions();
      if (!results.empty()) {
        assert(results.size() == 1 && results[0].future_id == id);
        completed = results[0];
        got = true;
      } else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(got && completed.ok == (mode == 0 || mode == 2));
    if (completed.ok) {
      f.check(f.read(c));
      auto n = c.metadata_stats()["registrations"];
      f.check(f.read(c));
      assert(c.metadata_stats()["registrations"] == n);
      fd = ::open(segment.c_str(), O_RDONLY);
      char hole = 1;
      assert(pread(fd, &hole, 1, 3 * 262144) == 1 && hole == 0);
      ::close(fd);
      // A store worker must reset its previous slot offset before returning
      // to a legacy object file (including compatibility rollback writes).
      auto legacy =
          c.submit_batch_set({wire_key(99)}, {input[0]}, {f.sizes[0]}, 0);
      got = false;
      for (int i = 0; i < 5000 && !got; ++i) {
        auto results = c.drain_completions();
        if (!results.empty()) {
          assert(results.size() == 1 && results[0].future_id == legacy &&
                 results[0].ok);
          got = true;
        } else
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      assert(got && std::filesystem::file_size(filename(f.path, 99)) == 229376);
      std::vector<char> saved(229376);
      std::ifstream in(filename(f.path, 99), std::ios::binary);
      in.read(saved.data(), saved.size());
      assert(saved == f.data[0]);
    }
    c.close();
    pass(
        "slot offsets / persistent FD / write failure drains / interrupted "
        "write");
  }
  {
    reset();
    Fixture f(std::vector<size_t>(4, 8));
    auto count_fds = [] {
      size_t count = 0;
      for (const auto& entry :
           std::filesystem::directory_iterator("/proc/self/fd")) {
        (void)entry;
        ++count;
      }
      return count;
    };
    const auto before = count_fds();
    Hf3fsConnector c(f.path, f.path, 1, 16, 0, -1, 128, 200, false, {}, 0, 0,
                     "", 16, true);
    assert(count_fds() > before + 256);
    f.check(f.read(c));  // New sharded reader also reads original flat objects.
    c.close();
    assert(count_fds() ==
           before);  // close(), not just destruction, releases directories.
    c.close();
    assert(count_fds() == before);
  }
  pass("close drains sharded directory handles before connector destruction");
  std::cout << tests << " test groups passed\n";
}
