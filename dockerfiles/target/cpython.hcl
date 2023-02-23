// Deps: utils, llvm, linker
// Deps: source_getter, cpp_builder

variable "CPYTHON_DIRS" {
    default = {
        source_dir = "/opt/source/cpython"
        build_dir = "/opt/build/cpython"
        install_dir = "/opt/cpython"
    }
}

variable "CPYTHON_ENV_VARS" {
    default = {
        // Path
        PATH = "${CPYTHON_DIRS.install_dir}/bin:$PATH"

        // Python
        PYTHONHOME = "${CPYTHON_DIRS.install_dir}"
        PYTHONPATH = "${CPYTHON_DIRS.install_dir}/lib/python3.10/site-packages"

        // Library
        LIBRARY_PATH = "${CPYTHON_DIRS.install_dir}/lib:$LIBRARY_PATH"
        LD_LIBRARY_PATH = "${CPYTHON_DIRS.install_dir}/lib:$LD_LIBRARY_PATH"

        // Include
        CPATH = "${CPYTHON_DIRS.install_dir}/include:$CPATH"
    }
}

variable "CPYTHON_COMPILE_FLAGS" {
    default = {
        CFLAGS = "-std=c17 ${llvm_compiler_flags({})}"
        CXXFLAGS = "-std=c++20 ${llvm_compiler_flags({})}"
        LDFLAGS = "${linker_flags({})} ${llvm_compiler_flags({})}"
    }
}

variable "CPYTHON_CONFIGURE_PHASE_FLAGS" {
    default = {
        "--prefix" = "${CPYTHON_DIRS.install_dir}"
        "--enable-ipv6" = "yes"
        "--enable-optimizations" = true
        "--with-computed-gotos" = true
        "--with-lto" = "thin"
        "--with-pymalloc" = true
        "--with-system-expat" = true
        "ax_cv_c_float_words_bigendian" = "no"
    }
}

variable "CPYTHON_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            MOLD_INSTALLER_ARGS
        ]
        unpack_phase = [{
            context = "cpython_source"
            dest = CPYTHON_DIRS.source_dir
        }]

        patch_phase = [
            // Patch and copy the source
            // Some things don't like the plus at the end so we remove it.
            "sed -i 's/3.10.9+/3.10.9/g' ${CPYTHON_DIRS.source_dir}/Include/patchlevel.h"
        ]

        configure_phase = [
            utils_export(CPYTHON_COMPILE_FLAGS),
            "mkdir -p ${CPYTHON_DIRS.build_dir}",
            "cd ${CPYTHON_DIRS.build_dir}",
            "${CPYTHON_DIRS.source_dir}/configure ${utils_format_flags({
                flags = CPYTHON_CONFIGURE_PHASE_FLAGS
            })}"
        ]
        build_phase = [
            utils_export(CPYTHON_COMPILE_FLAGS),
            "cd ${CPYTHON_DIRS.build_dir}",
            "make -j",
        ]
        install_phase = flatten([
            utils_export(CPYTHON_COMPILE_FLAGS),
            "cd ${CPYTHON_DIRS.build_dir}",
            "make install -j",
            "ln -s ${CPYTHON_DIRS.install_dir}/bin/python3 ${CPYTHON_DIRS.install_dir}/bin/python",
            utils_export(CPYTHON_ENV_VARS),
            pip_install({
                packages = {
                    pip = "23.0"
                    setuptools = "67.1.0"
                    wheel = "0.38.4"
                }
            })
        ])
    }
}

variable "CPYTHON_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "cpython_install"
                source = CPYTHON_DIRS.install_dir
                dest = CPYTHON_DIRS.install_dir
            }
        ]
        env_vars = CPYTHON_ENV_VARS
    }
}


variable "CPYTHON_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "cpython"
                source = CPYTHON_DIRS.install_dir
                dest = CPYTHON_DIRS.install_dir
            }
        ]
        env_vars = CPYTHON_ENV_VARS
    }
}

target "cpython_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "python"
        repo = "cpython"
        rev = "c3dd95a669030ff81f5e841d181110cdfd78e542"
    })
}

target "cpython_install" {
    contexts = merge(
        {base = "target:cpp_builder"},
        utils_get_unpack_phase_contexts(CPYTHON_BUILDER_ARGS)
    )
    dockerfile-inline = utils_builder(CPYTHON_BUILDER_ARGS)
}

target "cpython" {
    tags = ["connorbaker01/cpython:latest"]
    contexts = utils_get_unpack_phase_contexts(CPYTHON_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(CPYTHON_DISTRIBUTER_ARGS)
}