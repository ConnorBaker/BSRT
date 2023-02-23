// Deps: utils, apt

target "cpp_builder" {
    tags = ["connorbaker01/cpp_builder:latest"]
    dockerfile-inline = utils_unlines([
        utils_image({
            from = "docker.io/ubuntu:jammy-20230126"
            sha256 = "c985bc3f77946b8e92c9a3648c6f31751a7dd972e06604785e47303f4ad47c4c"
        }),

        // Basic environment variables
        utils_env({
            DEBIAN_FRONTEND = "noninteractive"
            LANG = "C.UTF-8"
            LC_ALL = "C.UTF-8"
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
                install = [
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
