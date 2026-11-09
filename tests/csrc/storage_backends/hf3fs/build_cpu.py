# SPDX-License-Identifier: Apache-2.0
"""Build CPU utility and HF3FS extensions for isolated source-tree tests."""

# Standard
from pathlib import Path
import os

# Third Party
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).resolve().parents[4]
common = root / "csrc/lmcache_native"
hf = root / "csrc/storage_backends/hf3fs"
include = os.environ["HF3FS_INCLUDE_DIR"]
library = os.environ["HF3FS_LIB_DIR"]
setup(
    name="lmcache-hf3fs-cpu-test",
    version="0.0.0",
    ext_modules=[
        CppExtension(
            "lmcache.lmcache_native",
            sources=[
                str(common / n)
                for n in (
                    "bitmap.cpp",
                    "fold.cpp",
                    "periodic_event_notifier.cpp",
                    "pybind.cpp",
                    "ttl_lock.cpp",
                    "utils.cpp",
                )
            ],
            include_dirs=[str(common), str(root / "csrc")],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        CppExtension(
            "lmcache.lmcache_hf3fs",
            sources=[
                str(hf / n)
                for n in ("pybind.cpp", "connector.cpp", "hf3fs_absl_init.cpp")
            ],
            include_dirs=[str(hf.parent), str(hf), include],
            library_dirs=[library, "/usr/lib/x86_64-linux-gnu"],
            libraries=[
                "hf3fs_api_shared",
                "absl_hash",
                "absl_city",
                "absl_int128",
                "absl_raw_hash_set",
            ],
            runtime_library_dirs=[library, "/usr/lib/x86_64-linux-gnu"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
