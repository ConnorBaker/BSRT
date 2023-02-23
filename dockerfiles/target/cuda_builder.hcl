// Deps: utils, apt

target "cuda_builder" {
    tags = ["connorbaker01/cuda_builder:latest"]
    dockerfile-inline = utils_unlines([
        utils_image({
            from = "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04"
            sha256 = "164e02e3505943c7117bc33e2458935572a00a3912f7c432794ad846efa502ef"
        }),

        // Basic environment variables
        utils_env({
            DEBIAN_FRONTEND = "noninteractive"
            LANG = "C.UTF-8"
            LC_ALL = "C.UTF-8"
            LIBRARY_PATH = "/usr/local/cuda/lib64:$LIBRARY_PATH"
            LD_LIBRARY_PATH = "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
            CUDA_HOME = "/usr/local/cuda"
            CUDA_MODULE_LOADING = "LAZY"
            CUDA_USE_STATIC_CUDA_RUNTIME = "ON"
        }),
        
        utils_run(concat(
            apt_install([
                "ca-certificates",
                "curl",
                "gnupg",
            ]),

            // Use the Enzu mirror for apt
            ["sed -i -e 's|http://archive.ubuntu.com|https://mirror.enzu.com|' /etc/apt/sources.list"],

            // Add CMake repo
            apt_add_repo({
                repo_name = "kitware"
                repo_url = "https://apt.kitware.com/ubuntu/"
                release = "jammy"
                keyring_name = "kitware-archive-keyring"
                pub_key_url = "https://apt.kitware.com/keys/kitware-archive-latest.asc"
            }),

            // Add Intel OneAPI repo
            apt_add_repo({
                repo_name = "oneAPI"
                repo_url = "https://apt.repos.intel.com/oneapi"
                release = "all"
                keyring_name = "oneapi-archive-keyring"
                pub_key_url = "https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
            }),

            // Add LLVM repo
            apt_add_repo({
                repo_name = "llvm"
                repo_url = "http://apt.llvm.org/jammy"
                release = "llvm-toolchain-jammy-16"
                keyring_name = "llvm-snapshot-keyring"
                pub_key_url = "https://apt.llvm.org/llvm-snapshot.gpg.key"
            }),


            apt_upgrade({
                unhold = [
                    "libcublas-12-0",
                    "libcublas-dev-12-0",
                    "libnccl-dev",
                    "libnccl2",
                ]
                install = [
                    // Install the latest cudnn and other packages
                    "libcudnn8",
                    "libcudnn8-dev",
                    // NOTE: nvidia-cuda-dev provides /usr/lib/x86_64-linux-gnu/liblapack_static.a
                    "nvidia-cuda-dev",

                    // General utilities
                    "git",
                    "lcov",
                    "libbz2-dev",
                    "libffi-dev",
                    "libgdbm-compat-dev",
                    "libgdbm-dev",
                    "libjpeg-turbo8-dev",
                    "liblzma-dev",
                    "libncurses5-dev",
                    "libnuma-dev",
                    "libopenblas-dev",
                    "libpng-dev",
                    "libprotobuf-dev",
                    "libreadline6-dev",
                    "libsqlite3-dev",
                    "libssl-dev",
                    "lzma-dev",
                    "ninja-build",
                    "numactl",
                    "pkg-config",
                    "protobuf-compiler",
                    "tk-dev",
                    "uuid-dev",
                    "zlib1g-dev",
                    "zstd",

                    // Install CMake
                    "cmake",

                    // Intel OneAPI MKL
                    "intel-oneapi-mkl-devel-2023.0.0",
                    "intel-oneapi-openmp-2023.0.0",
                    "intel-oneapi-runtime-openmp",

                    // Install LLVM
                    "clang-16",
                    "libclang-rt-16-dev",
                    "libomp-16-dev",
                    "libomp5-16",
                    "lld-16",
                    "lldb-16"
                ]
            }),
            
            // NOTE: nvidia-cuda-dev puts liblapack* in the wrong directory! Link them to the cuda 
            //  lib dir. We must symlink liblapack.a, liblapack.so, liblapack.so.3, and 
            //  liblapack_static.a.
            // 
            // TODO: Report bug to pytorch. Seems like with newer CUDA docker images 
            //  nvidia-cuda-dev puts the libs in /usr/lib/x86_64-linux-gnu instead of /usr/local/
            //  cuda/lib64. See symbolic links in base image for example.
            // 
            //  Error thrown: #0 76.50 ninja: error: '/usr/local/cuda/lib64/liblapack_static.a', 
            //  needed by 'lib/libtorch_cuda_linalg.so', missing and no known rule to make it
            // 
            //  Relevant links:
            //  https://github.com/pytorch/pytorch/blob/ee2729890c32c77ec1948d81ad1080585e232468/aten/src/ATen/CMakeLists.txt#L448
            //  https://github.com/pytorch/pytorch/blob/e790281a85fe3693fc1d38bf0e2c6e874d5e10b0/caffe2/CMakeLists.txt#L960
            [
                "ln -s /usr/lib/x86_64-linux-gnu/liblapack.a /usr/local/cuda/lib64/liblapack.a",
                "ln -s /usr/lib/x86_64-linux-gnu/liblapack.so /usr/local/cuda/lib64/liblapack.so",
                "ln -s /usr/lib/x86_64-linux-gnu/liblapack.so.3 /usr/local/cuda/lib64/liblapack.so.3",
                "ln -s /usr/lib/x86_64-linux-gnu/liblapack_static.a /usr/local/cuda/lib64/liblapack_static.a",
            ],
        )),

        utils_env({
            // Intel OneAPI environment variables
            MKLROOT = "/opt/intel/oneapi/mkl/latest"

            // LLVM environment variables
            LLVM_INSTALL_DIR = "/usr/lib/llvm-16"

            // Put LLVM binaries in the PATH
            PATH = "/usr/lib/llvm-16/bin:$PATH"

            // Put LLVM libraries in the LIBRARY_PATH
            LIBRARY_PATH = "/usr/lib/llvm-16/lib:$LIBRARY_PATH"
            // We can use LD_LIBRARY_PATH if we need them available at runtime
            
            // Put LLVM includes in the CPATH (both C and C++)
            CPATH = "/usr/lib/llvm-16/include:$CPATH"

            CC = "clang"
            CXX = "clang++"
            LD = "lld"

            // CUDA-related environment variables
            CUDAHOSTCXX = "clang++"

            AR = "llvm-ar"
            AS = "llvm-as"
            NM = "llvm-nm"
            OBJCOPY = "llvm-objcopy"
            OBJDUMP = "llvm-objdump"
            RANLIB = "llvm-ranlib"
            READELF = "llvm-readelf"
            STRIP = "llvm-strip"
        })
    ])
}

// Build the image
target "runner" {
    contexts = merge(
        {base = "target:cuda_builder"},
        
    )
}