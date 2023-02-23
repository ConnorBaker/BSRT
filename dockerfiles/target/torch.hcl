// Deps: utils, llvm, linker, cmake

variable "TORCH_DIRS" {
    default = {
        source_dir = "/opt/source/torch",
        build_dir = "/opt/build/torch",
        install_dir = "/opt/torch",
    }
}

variable "TORCH_ENV_VARS" {
    default = {
        TORCH_HOME = TORCH_DIRS.install_dir,
        PYTHONPATH = "${TORCH_DIRS.install_dir}:$PYTHONPATH",
    }
}


variable "TORCH_BASE_FLAGS" {
    default = {
        // TODO: Linking with LLD
        LD = "lld"

        // TODO: Why do we need -fPIC specified for libnnpack?
        //  Bug in PyTorch: https://github.com/pytorch/pytorch/pull/93896
        COMMON_FLAGS = "${llvm_compiler_flags({
            force_emit_vtables = false
            lto = false
            whole_program_vtables = false
            split_lto_unit = false
        })} -fPIC"
        
        // NOTE: We can pass LTO_FLAGS to the linker so we can use the LTO objects from Magma, 
        //  but we can't use them in COMMON_FLAGS because we can't compile torch with LTO.
        LINK_FLAGS = "${linker_flags({})} ${llvm_compiler_flags({})}"
    }
}

variable "TORCH_CMAKE_FLAGS_EXPORT_BASH_CMDS" {
    default = [
        // Set up CUDA flags
        // TODO: Why do we need -shared specified for CMake's CUDA compiler check to succeed?
        //  Bug in PyTorch: https://github.com/pytorch/pytorch/pull/93896
        "export XLINKER_FLAGS=\"$${LINK_FLAGS//-Wl,/} -fPIC -shared\"",
        "export XLINKER_FLAGS=\"$${XLINKER_FLAGS//-march=native/}\"",
        "export XLINKER_FLAGS=\"$${XLINKER_FLAGS//--pipe/}\"",
        "export XLINKER_FLAGS=\"$${XLINKER_FLAGS// /,}\"",
        "export XCOMPILER_FLAGS=\"$${COMMON_FLAGS// /,}\"",
        "export CUDA_FLAGS=\"--std=c++17 --allow-unsupported-compiler -O3 --threads=0 --extra-device-vectorization\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xfatbin=--compress-all\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xcompiler=$XCOMPILER_FLAGS\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xlinker=$XLINKER_FLAGS\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xnvlink=--use-host-info\"",

        // Base CMake settings
        "export BUILD_SHARED_LIBS=ON",
        "export CMAKE_BUILD_TYPE=Release",
        "export CMAKE_VERBOSE_MAKEFILE=ON",

        // Set C standard and flags
        "export CMAKE_C_STANDARD=17",
        "export CMAKE_C_STANDARD_REQUIRED=ON",
        "export CMAKE_C_EXTENSIONS=ON",
        "export CMAKE_C_FLAGS=\"$COMMON_FLAGS\"",

        // Set C++ standard and flags
        "export CMAKE_CXX_STANDARD=17",
        "export CMAKE_CXX_STANDARD_REQUIRED=ON",
        "export CMAKE_CXX_EXTENSIONS=ON",
        "export CMAKE_CXX_FLAGS=\"$COMMON_FLAGS\"",

        // Set linker flags
        "export CMAKE_EXE_LINKER_FLAGS=\"$LINK_FLAGS\"",
        "export CMAKE_MODULE_LINKER_FLAGS=\"$LINK_FLAGS\"",
        "export CMAKE_SHARED_LINKER_FLAGS=\"$LINK_FLAGS\"",

        // LTO policies
        "export CMAKE_POLICY_DEFAULT_CMP0069=NEW",
        "export CMAKE_POLICY_DEFAULT_CMP0105=NEW",
        "export CMAKE_POLICY_DEFAULT_CMP0138=NEW",

        // LTO flags
        // TODO: Since https://github.com/pytorch/pytorch/pull/93388 landed, we should be able 
        //  to build with LTO!
        "export CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
        "export CMAKE_POSITION_INDEPENDENT_CODE=ON",

        // CUDA LTO
        // NOTE: Make sure RESOLVE_DEVICE_SYMBOLS is OFF!
        //       https://gitlab.kitware.com/cmake/cmake/-/issues/22225
        "export CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS=OFF",
        "export CMAKE_CUDA_SEPARABLE_COMPILATION=OFF",
        "export CUDA_SEPARABLE_COMPILATION=OFF",

        "export CMAKE_CUDA_ARCHITECTURES=89-real",
        "export CMAKE_CUDA_FLAGS=\"$CUDA_FLAGS\"",
        "export CMAKE_CUDA_HOST_COMPILER=\"$CXX\"",
        "export CMAKE_CUDA_STANDARD=17",
    ]
}

variable "TORCH_FLAGS" {
    default = {
        // Torch-specific flags
        ATEN_NO_TEST = "ON"
        ATEN_STATIC_CUDA = "ON"
        BLAS = "MKL"
        BUILD_CUSTOM_PROTOBUF = "OFF"
        BUILD_TEST = "OFF"
        INSTALL_TEST = "OFF"
        INTEL_MKL_DIR = "$MKLROOT"
        NCCL_ROOT = "/usr"
        Protobuf_USE_STATIC_LIBS = "ON"
        TH_BINARY_BUILD = "ON"
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = "1"
        TORCH_CUDA_ARCH_LIST = "8.9"
        // TODO: Do we need to escape it here? Escapes should be handled in the function which turns this into a string.
        // TORCH_NVCC_FLAGS = "\"$CUDA_FLAGS\""
        TORCH_NVCC_FLAGS = "$CUDA_FLAGS"
        USE_CUDA_STATIC_LINK = "ON"
        USE_CUDNN = "ON"
        USE_CUPTI_SO = "OFF"
        USE_EXPERIMENTAL_CUDNN_V8_API = "ON"
        USE_FLASH_ATTENTION = "ON"
        USE_GLOO = "OFF"
        USE_KINETO = "ON"
        USE_MKLDNN = "OFF"
        USE_NCCL = "ON"
        USE_STATIC_CUDNN = "OFF"
        USE_STATIC_MKL = "ON"
        USE_STATIC_NCCL = "ON"
        USE_SYSTEM_NCCL = "ON"
        USE_SYSTEM_PYBIND11 = "ON"
    }
}

variable "TORCH_PIP_INSTALL_ARGS" {
    default = {
        packages = {
            "${TORCH_DIRS.source_dir}" = null
        }
        flags = {
            no_deps = true
            target = TORCH_DIRS.install_dir
        }
    }
}

variable "TORCH_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            MOLD_INSTALLER_ARGS,
            CPYTHON_INSTALLER_ARGS,
            COMMON_PYTHON_PACKAGES_INSTALLER_ARGS,
            MAGMA_INSTALLER_ARGS
        ]
        
        unpack_phase = [
            {
                context = "torch_source"
                dest = TORCH_DIRS.source_dir
            },
        ]

        patch_phase = [
            "sed -i -e 's/Werror=cast/Wcast/g' ${TORCH_DIRS.source_dir}/CMakeLists.txt",
            // TODO: Report bug to PyTorch. Apparently the version of the headers in /usr/include 
            //  does not match the version of nccl in /usr/lib/x86_64-linux-gnu.
            // 
            //  The header has NCCL_MAJOR=2, NCCL_MINOR=16, NCCL_PATCH=5, NCCL_SUFFIX="", 
            //  NCCL_VERSION=2.16.5, but the static library has "2.16.5+cuda12.0".
            // 
            //  The fix is to remove the version check in FindNCCL.cmake.
            "sed -i '74,81d' ${TORCH_DIRS.source_dir}/cmake/Modules/FindNCCL.cmake",
        ]

        pre_install_phase = flatten([
            utils_export(TORCH_BASE_FLAGS),
            TORCH_CMAKE_FLAGS_EXPORT_BASH_CMDS,
            utils_export(TORCH_FLAGS)
        ])
        install_phase = pip_install(TORCH_PIP_INSTALL_ARGS)
    }
}


variable "TORCH_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "torch_install"
                source = TORCH_DIRS.install_dir
                dest = TORCH_DIRS.install_dir
            }
        ]
        env_vars = TORCH_ENV_VARS
    }
}


variable "TORCH_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "torch"
                source = TORCH_DIRS.install_dir
                dest = TORCH_DIRS.install_dir
            }
        ]
        env_vars = TORCH_ENV_VARS
    }
}


target "torch_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "pytorch"
        repo = "pytorch"
        rev = "f84f89b1c3f2bc74512e7a7b05ae6185164a9b3e"
    })
}

// Added super init to Module
// TODO: Can we set this to true in dataclass derivations of Module to avoid needing to 
//  call super in __post__init__?
//  See: https://github.com/pytorch/pytorch/pull/91819.

target "torch_install" {
    contexts = merge(
        {base = "target:cuda_builder"},
        utils_get_unpack_phase_contexts(TORCH_BUILDER_ARGS)
    )
    dockerfile-inline = utils_builder(TORCH_BUILDER_ARGS)
}

target "torch" {
    tags = ["connorbaker01/torch:latest"]
    contexts = utils_get_unpack_phase_contexts(TORCH_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(TORCH_DISTRIBUTER_ARGS)
}