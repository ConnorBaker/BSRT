// Deps: utils, pip, llvm, linker, cmake

target "pybind11_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "pybind"
        repo = "pybind11"
        rev = "3efe9d4cb5d7314faee722205e560b8b932aed9e"
    })
}

target "pillow_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "uploadcare"
        repo = "pillow-simd"
        rev = "9.2.0"
    })
}


variable "COMMON_PYTHON_PACKAGES_DIRS" {
    default = {
        source_dir = "/opt/source/common_python_packages"
        build_dir = "/opt/build/common_python_packages"
        install_dir = "/opt/common_python_packages"
    }
}

variable "COMMON_PYTHON_PACKAGES_ENV_VARS" {
    default = {
        // Path
        PATH = "${COMMON_PYTHON_PACKAGES_DIRS.install_dir}/bin:$PATH"

        // Python
        PYTHONPATH = "${COMMON_PYTHON_PACKAGES_DIRS.install_dir}:$PYTHONPATH"
        
        pybind11_DIR = "${COMMON_PYTHON_PACKAGES_DIRS.install_dir}/pybind11"
    }
}

variable "COMMON_PYTHON_PACKAGES_COMPILE_FLAGS" {
    default = {
        CFLAGS = "-std=c17 ${llvm_compiler_flags({})}"
        CXXFLAGS = "-std=c++20 ${llvm_compiler_flags({})}"
        LDFLAGS = "${linker_flags({})} ${llvm_compiler_flags({})}"
    }
}


variable "COMMON_PYTHON_PACKAGES_PIP_INSTALL_ARGS" {
    default = {
        pre_install = [
            utils_export(COMMON_PYTHON_PACKAGES_COMPILE_FLAGS),
        ]
        packages = {
            // Core dependencies
            packaging = "23.0"
            
            // NOTE: numpy is a hassle to compile with LTO, so we use the precompiled 
            //  version.
            numpy = "1.21.4"
            requests = "2.28.2"
            typing_extensions = "4.4.0"

            "${COMMON_PYTHON_PACKAGES_DIRS.source_dir}/pillow" = null
            "${COMMON_PYTHON_PACKAGES_DIRS.source_dir}/pybind11" = null

            // PyTorch dependencies
            // NOTE: Need Jinja2 for some templating done by torch.compile
            // TODO: Check if still the case; if it is, see if it's included in 
            //  PyTorch now.
            Cython = "0.29.33"
            Jinja2 = "3.1.2"
            networkx = "3.0"
            pyyaml = "6.0"
            sympy = "1.11.1"
            
            // Triton dependencies
            filelock = "3.9.0"
            lit = "15.0.7"
        }
        flags = {
            target = COMMON_PYTHON_PACKAGES_DIRS.install_dir
        }
    }
}

variable "COMMON_PYTHON_PACKAGES_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            MOLD_INSTALLER_ARGS,
            CPYTHON_INSTALLER_ARGS
        ]
        unpack_phase = [
            {
                context = "pybind11_source"
                dest = "${COMMON_PYTHON_PACKAGES_DIRS.source_dir}/pybind11"
            },
            {
                context = "pillow_source"
                dest = "${COMMON_PYTHON_PACKAGES_DIRS.source_dir}/pillow"
            },
        ]

        install_phase = pip_install(COMMON_PYTHON_PACKAGES_PIP_INSTALL_ARGS)
    }
}

variable "COMMON_PYTHON_PACKAGES_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "common_python_packages_install"
                source = COMMON_PYTHON_PACKAGES_DIRS.install_dir
                dest = COMMON_PYTHON_PACKAGES_DIRS.install_dir
            }
        ]
        env_vars = COMMON_PYTHON_PACKAGES_ENV_VARS
    }
}


variable "COMMON_PYTHON_PACKAGES_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "common_python_packages"
                source = COMMON_PYTHON_PACKAGES_DIRS.install_dir
                dest = COMMON_PYTHON_PACKAGES_DIRS.install_dir
            }
        ]
        env_vars = COMMON_PYTHON_PACKAGES_ENV_VARS
    }
}

target "common_python_packages_install" {
    contexts = merge(
        {base = "target:cpp_builder"},
        utils_get_unpack_phase_contexts(COMMON_PYTHON_PACKAGES_BUILDER_ARGS)
    )
    dockerfile-inline = utils_builder(COMMON_PYTHON_PACKAGES_BUILDER_ARGS)
}

target "common_python_packages" {
    tags = ["connorbaker01/common_python_packages:latest"]
    contexts = utils_get_unpack_phase_contexts(COMMON_PYTHON_PACKAGES_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(COMMON_PYTHON_PACKAGES_DISTRIBUTER_ARGS)
}