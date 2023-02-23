// Deps: utils, llvm, linker, cmake

variable "MAGMA_DIRS" {
    default = {
        source_dir = "/opt/source/magma"
        build_dir = "/opt/build/magma"
        install_dir = "/opt/magma"
    }
}

variable "MAGMA_ENV_VARS" {
    default = {
        MAGMA_HOME = MAGMA_DIRS.install_dir

        // Library
        LIBRARY_PATH = "${MAGMA_DIRS.install_dir}/lib:$LIBRARY_PATH"
        LD_LIBRARY_PATH = "${MAGMA_DIRS.install_dir}/lib:$LD_LIBRARY_PATH"

        // Include
        CPATH = "${MAGMA_DIRS.install_dir}/include:$CPATH"
    }
}

variable "MAGMA_CMAKE_CONFIGURE_PHASE_FLAGS" {
    default = {
        cmake = {
            c_extensions = true
            c_standard = 17
            cxx_extensions = true
            cxx_standard = 17
            build_shared_libs = false
            extra_flags = {
                // CUDA LTO
                // NOTE: Make sure RESOLVE_DEVICE_SYMBOLS is OFF!
                //       https://gitlab.kitware.com/cmake/cmake/-/issues/22225
                CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS="OFF"
                CMAKE_CUDA_SEPARABLE_COMPILATION="ON"
                CMAKE_CUDA_ARCHITECTURES="89-real"

                CMAKE_CUDA_HOST_COMPILER="$CXX"
                CMAKE_CUDA_STANDARD="17"
                CMAKE_CUDA_FLAGS="$CUDA_FLAGS"

                // Magma-specific flags
                GPU_TARGET="Ampere"
                MAGMA_ENABLE_CUDA="ON"
                USE_FORTRAN="OFF"
            }
        }
    }
}

variable "MAGMA_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            MOLD_INSTALLER_ARGS,
            CPYTHON_INSTALLER_ARGS,
        ]
        unpack_phase = [
            {
                context = "magma_source"
                dest = MAGMA_DIRS.source_dir
            },
            {
                context = "magma_source_patches"
                dest = MAGMA_DIRS.source_dir
            }
        ]
        patch_phase = [
            "cd ${MAGMA_DIRS.source_dir}",
            "git apply magma_remove_tests.diff",
            // TODO: Bug report for magma so they can use newer standards
            "git apply magma_use_std-cpp17_std-c17.diff",
        ]
        pre_configure_phase = [
            "cd ${MAGMA_DIRS.source_dir}",

            "echo BACKEND = cuda >> make.inc",
            "echo FORT = false >> make.inc",
            "echo GPU_TARGET = Ampere >> make.inc",
            
            "export LINK_FLAGS=\"${linker_flags({})}\"",
            "export XLINKER_FLAGS=\"$${LINK_FLAGS//-Wl,/}\"",
            "export XLINKER_FLAGS=\"$${XLINKER_FLAGS// /,}\"",
            // NOTE: Update the link flags because we cannot include the compiler flags 
            //  earlier since they would pollute XLINKER_FLAGS.
            "export LINK_FLAGS=\"$LINK_FLAGS ${llvm_compiler_flags({})}\"",
            
            "export COMMON_FLAGS=\"${llvm_compiler_flags({})}\"",

            "export CUDA_FLAGS=\"--std=c++17 --allow-unsupported-compiler -O3 --threads=0 --extra-device-vectorization\"",
            "export CUDA_FLAGS=\"$CUDA_FLAGS -Xfatbin=--compress-all\"",
            "export CUDA_FLAGS=\"$CUDA_FLAGS -Xcompiler=$${COMMON_FLAGS// /,}\"",
            "export CUDA_FLAGS=\"$CUDA_FLAGS -Xlinker=$XLINKER_FLAGS\"",
            "export CUDA_FLAGS=\"$CUDA_FLAGS -Xnvlink=--use-host-info\"",
            
            "make generate -j",
        ]
        configure_phase = merge(
            MAGMA_DIRS,
            {flags = MAGMA_CMAKE_CONFIGURE_PHASE_FLAGS}
        )
        build_phase = MAGMA_DIRS
        install_phase = MAGMA_DIRS
    }
}


variable "MAGMA_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "magma_install"
                source = MAGMA_DIRS.install_dir
                dest = MAGMA_DIRS.install_dir
            }
        ]
        env_vars = MAGMA_ENV_VARS
    }
}

variable "MAGMA_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "magma"
                source = MAGMA_DIRS.install_dir
                dest = MAGMA_DIRS.install_dir
            }
        ]
        env_vars = MAGMA_ENV_VARS
    }
}

target "magma_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "bitbucket.org"
        owner = "icl"
        repo = "magma"
        rev = "f3a24d3fd0538bc20931b3c01fe9964ca41a7634"
    })
}

target "magma_source_patches" {
    dockerfile-inline = <<-EOF
        FROM scratch
        COPY ./dockerfiles/patches/magma_remove_tests.diff ./dockerfiles/patches/magma_use_std-cpp17_std-c17.diff .
    EOF
}

target "magma_install" {
    contexts = merge(
        {base = "target:cuda_builder"},
        utils_get_unpack_phase_contexts(MAGMA_BUILDER_ARGS)
    )
    dockerfile-inline = cmake_builder(MAGMA_BUILDER_ARGS)
}

target "magma" {
    tags = ["connorbaker01/magma:latest"]
    contexts = utils_get_unpack_phase_contexts(MAGMA_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(MAGMA_DISTRIBUTER_ARGS)
}

