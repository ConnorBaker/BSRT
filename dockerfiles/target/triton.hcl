// Deps: utils, pip, llvm, linker

variable "TRITON_DIRS" {
    default = {
        source_dir = "/opt/source/triton"
        build_dir = "/opt/build/triton"
        install_dir = "/opt/triton"
    }
}

variable "TRITON_ENV_VARS" {
    default = {
        TRITON_HOME = TRITON_DIRS.install_dir
        PYTHONPATH = "${TRITON_DIRS.install_dir}:$PYTHONPATH"
    }
}

variable "TRITON_BASE_FLAGS" {
    default = {
        // TODO: whole_program_vtables causes the following: 
        //  error: cannot be converted to LLVM IR: missing 
        //  `LLVMTranslationDialectInterface` registration for dialect for op: arith.
        //  constant.
        COMMON_FLAGS = llvm_compiler_flags({
            whole_program_vtables = false
        })
        LINK_FLAGS = "${linker_flags({})} ${llvm_compiler_flags({
            whole_program_vtables = false
        })}"
    }
}

variable "TRITON_COMPILE_FLAGS" {
    default = {
        CFLAGS = "-std=gnu17 $COMMON_FLAGS"
        CXXFLAGS = "-std=gnu++17 $COMMON_FLAGS"
        LDFLAGS = "$LINK_FLAGS"
    }
}

variable "TRITON_PIP_INSTALL_ARGS" {
    default = {
        packages = {
            "${TRITON_DIRS.source_dir}/python" = null
        }
        flags = {
            // NOTE: Must use editable installation to avoid cyclilc imports.
            // editable = true
            // NOTE: Use --no-deps to ensure triton uses our torch installation instead of the 
            // latest stable version.
            no_deps = true
            target = TRITON_DIRS.install_dir
            verbose = true
        }
    }
}

variable "TRITON_BUILDER_ARGS" {
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
                context = "triton_source"
                dest = TRITON_DIRS.source_dir
            },
            {
                context = "triton_llvm_source"
                dest = "/opt/llvm14"
            },
        ]

        pre_install_phase = flatten([
            utils_export(TRITON_BASE_FLAGS),
            utils_export(TRITON_COMPILE_FLAGS),
            "export pybind11_SYSPATH=$pybind11_DIR",
            "export LLVM_SYSPATH=/opt/llvm14",

        ])
        install_phase = pip_install(TRITON_PIP_INSTALL_ARGS)
    }
}

target "triton_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "openai"
        repo = "triton"
        rev = "a9d1935e795cf28aa3c3be8ac5c14723e6805de5"
    })
}

target "triton_llvm_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_tarball({
        url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
    })
}

variable "TRITON_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "triton_install"
                source = TRITON_DIRS.install_dir
                dest = TRITON_DIRS.install_dir
            },
        ]
        env_vars = TRITON_ENV_VARS
    }
}

variable "TRITON_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "triton"
                source = TRITON_DIRS.install_dir
                dest = TRITON_DIRS.install_dir
            }
        ]
        env_vars = TRITON_ENV_VARS
    }
}


// utils_run([
//     // NOTE: Triton expects a specific version of pybind11. We patch setup.py to use the 
//     //  version we have installed and symlink it.
//     "sed -i 's/pybind11-2.10.0/pybind11/g' /triton/python/setup.py",
// ]),

target "triton_install" {
    contexts = merge(
        { base = "target:cuda_builder" },
        utils_get_unpack_phase_contexts(TRITON_BUILDER_ARGS),
    )
    dockerfile-inline = utils_builder(TRITON_BUILDER_ARGS)
}

target "triton" {
    tags = ["connorbaker01/triton:latest"]
    contexts = utils_get_unpack_phase_contexts(TRITON_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(TRITON_DISTRIBUTER_ARGS)
}
