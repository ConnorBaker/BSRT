// Deps: utils, cmake
// Deps: source_getter, cpp_builder, mimalloc

variable "MOLD_DIRS" {
    default = {
        source_dir = "/opt/source/mold"
        build_dir = "/opt/build/mold"
        install_dir = "/opt/mold"
    }
}

variable "MOLD_ENV_VARS" {
    default = {
        // Path
        PATH = "${MOLD_DIRS.install_dir}/bin:$PATH"

        // LD
        LD = "mold"

        // Library
        LIBRARY_PATH = "${MOLD_DIRS.install_dir}/lib:$LIBRARY_PATH"
        LD_LIBRARY_PATH = "${MOLD_DIRS.install_dir}/lib:$LD_LIBRARY_PATH"
    }
}

variable "MOLD_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS
        ]
        unpack_phase = [{
            context = "mold_source"
            dest = MOLD_DIRS.source_dir
        }]
        configure_phase = merge(
            MOLD_DIRS,
            {
                flags = {
                    cmake = {
                        // Mold requires GNU extensions
                        c_extensions = true
                        cxx_extensions = true
                        cxx_standard = 20
                        extra_flags = {
                            MOLD_LTO = true
                            MOLD_USE_MIMALLOC = true
                            MOLD_USE_SYSTEM_MIMALLOC = true
                            ZSTD_LEGACY_SUPPORT = false
                        }
                    }
                }
            }
        )
        build_phase = MOLD_DIRS
        install_phase = MOLD_DIRS
    }
}

variable "MOLD_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "mold_install"
                source = MOLD_DIRS.install_dir
                dest = MOLD_DIRS.install_dir
            }
        ]
        env_vars = MOLD_ENV_VARS
    }
}

variable "MOLD_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "mold"
                source = MOLD_DIRS.install_dir
                dest = MOLD_DIRS.install_dir
            }
        ]
        env_vars = MOLD_ENV_VARS
    }
}

target "mold_source" {
    contexts = {
        base = "target:source_getter"
    }
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "rui314"
        repo = "mold"
        rev = "e50590043cb8d92fedfa48a08415511565fe2659"
    })
}


target "mold_install" {
    contexts = merge(
        {base = "target:cpp_builder"},
        utils_get_unpack_phase_contexts(MOLD_BUILDER_ARGS)
    )
    dockerfile-inline = cmake_builder(MOLD_BUILDER_ARGS)
}

target "mold" {
    tags = ["connorbaker01/mold:latest"]
    contexts = utils_get_unpack_phase_contexts(MOLD_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(MOLD_DISTRIBUTER_ARGS)
}

