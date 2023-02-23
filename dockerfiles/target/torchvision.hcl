// Deps: utils, llvm, linker

variable "TORCHVISION_DIRS" {
    default = {
        source_dir = "/opt/source/torchvision",
        build_dir = "/opt/build/torchvision",
        install_dir = "/opt/torchvision",
    }
}

variable "TORCHVISION_ENV_VARS" {
    default = {
        TORCHVISION_HOME = TORCHVISION_DIRS.install_dir,
        PYTHONPATH = "${TORCHVISION_DIRS.install_dir}:$PYTHONPATH",
    }
}

variable "TORCHVISION_BASE_FLAGS" {
    default = {
        COMMON_FLAGS = llvm_compiler_flags({})
        LINK_FLAGS = "${linker_flags({})} ${llvm_compiler_flags({})}"
    }
}

variable "TORCHVISION_FLAGS_EXPORT_BASH_CMDS" {
    default = [
        // Set up CUDA flags
        "export XLINKER_FLAGS=\"$${LINK_FLAGS//-Wl,/}\"",
        "export XLINKER_FLAGS=\"$${XLINKER_FLAGS// /,}\"",
        "export XCOMPILER_FLAGS=\"$${COMMON_FLAGS// /,}\"",
        "export CUDA_FLAGS=\"--std=c++17 --allow-unsupported-compiler -O3 --threads=0 --extra-device-vectorization\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xfatbin=--compress-all\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xcompiler=$XCOMPILER_FLAGS\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xlinker=$XLINKER_FLAGS\"",
        "export CUDA_FLAGS=\"$CUDA_FLAGS -Xnvlink=--use-host-info\"",

        // Set C standard and flags
        "export CFLAGS=\"-std=c17 $COMMON_FLAGS\"",

        // Set C++ standard and flags
        "export CXXFLAGS=\"-std=c++17 $COMMON_FLAGS\"",

        // Set linker flags
        "export LDFLAGS=\"$LINK_FLAGS\"",
    ]
}

variable "TORCHVISION_FLAGS" {
    default = {
        DEBUG = "0"
        FORCE_CUDA = "1"
        NVCC_FLAGS = "$CUDA_FLAGS"
        TORCH_CUDA_ARCH_LIST = "8.9"
        TORCHVISION_USE_FFMPEG = "0"
        TORCHVISION_USE_JPEG = "1"
        TORCHVISION_USE_NVJPEG = "0"
        TORCHVISION_USE_PNG = "1"
        TORCHVISION_USE_VIDEO_CODEC = "0"
    }
}

variable "TORCHVISION_PIP_INSTALL_ARGS" {
    default = {
        packages = {
            "${TORCHVISION_DIRS.source_dir}" = null
        }
        flags = {
            no_deps = true
            target = TORCHVISION_DIRS.install_dir
            verbose = true
        }
    }
}

variable "TORCHVISION_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            MOLD_INSTALLER_ARGS,
            CPYTHON_INSTALLER_ARGS,
            COMMON_PYTHON_PACKAGES_INSTALLER_ARGS,
            TORCH_INSTALLER_ARGS,
        ]

        unpack_phase = [
            {
                context = "torchvision_source"
                dest = TORCHVISION_DIRS.source_dir
            },
        ]
        
        patch_phase = [
            // Patch the source
            // TODO: PyTorch's dynamo doesn't support `copy` on lists.
            //  Bug in PyTorch: https://github.com/pytorch/pytorch/issues/93906
            "sed -i 's/shift_size.copy()/\\[_ for _ in shift_size\\]/g' ${TORCHVISION_DIRS.source_dir}/torchvision/models/swin_transformer.py",
        ]

        pre_install_phase = flatten([
            utils_export(TORCHVISION_BASE_FLAGS),
            TORCHVISION_FLAGS_EXPORT_BASH_CMDS,
            utils_export(TORCHVISION_FLAGS),
        ])
        install_phase = pip_install(TORCHVISION_PIP_INSTALL_ARGS)
    }
}

target "torchvision_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "pytorch"
        repo = "vision"
        rev = "135a0f9ea9841b6324b4fe8974e2543cbb95709a"
    })
}


variable "TORCHVISION_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "torchvision_install"
                source = TORCHVISION_DIRS.install_dir
                dest = TORCHVISION_DIRS.install_dir
            }
        ]
        env_vars = TORCHVISION_ENV_VARS
    }
}


variable "TORCHVISION_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "torchvision"
                source = TORCHVISION_DIRS.install_dir
                dest = TORCHVISION_DIRS.install_dir
            }
        ]
        env_vars = TORCHVISION_ENV_VARS
    }
}


target "torchvision_install" {
    contexts = merge(
        {base = "target:cuda_builder"},
        utils_get_unpack_phase_contexts(TORCHVISION_BUILDER_ARGS)
    )
    dockerfile-inline = utils_builder(TORCHVISION_BUILDER_ARGS)
}

target "torchvision" {
    tags = ["connorbaker01/torchvision:latest"]
    contexts = utils_get_unpack_phase_contexts(TORCHVISION_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(TORCHVISION_DISTRIBUTER_ARGS)
}