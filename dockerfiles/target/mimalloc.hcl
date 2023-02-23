// Deps: utils, cmake
// Deps: source_getter, cpp_builder

variable "MIMALLOC_DIRS" {
    default = {
        source_dir = "/opt/source/mimalloc"
        build_dir = "/opt/build/mimalloc"
        install_dir = "/opt/mimalloc"
    }
}

variable "MIMALLOC_ENV_VARS" {
    default = {
        // Preload
        LD_PRELOAD = "${MIMALLOC_DIRS.install_dir}/lib/libmimalloc.so:$LD_PRELOAD"

        // Library paths
        LIBRARY_PATH = "${MIMALLOC_DIRS.install_dir}/lib:$LIBRARY_PATH"
        LD_LIBRARY_PATH = "${MIMALLOC_DIRS.install_dir}/lib:$LD_LIBRARY_PATH"

        // Include paths
        CPATH = "${MIMALLOC_DIRS.install_dir}/include:$CPATH"
    }   
}

variable "MIMALLOC_BUILDER_ARGS" {
    default = {
        unpack_phase = [{
            context = "mimalloc_source"
            dest = MIMALLOC_DIRS.source_dir
        }]
        configure_phase = merge(
            MIMALLOC_DIRS,
            {
                flags = {
                    cmake = {
                        cxx_standard = 20
                        extra_flags = {
                            MI_BUILD_OBJECT = false
                            MI_BUILD_SHARED = true
                            MI_BUILD_STATIC = false
                            MI_BUILD_TESTS = false
                            MI_OVERRIDE = true
                        }
                    }
                }
            }
        )
        build_phase = MIMALLOC_DIRS
        install_phase = MIMALLOC_DIRS
    }
}

variable "MIMALLOC_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "mimalloc_install"
                source = MIMALLOC_DIRS.install_dir
                dest = MIMALLOC_DIRS.install_dir
            }
        ]
        env_vars = MIMALLOC_ENV_VARS
    }
}

variable "MIMALLOC_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "mimalloc"
                source = MIMALLOC_DIRS.install_dir
                dest = MIMALLOC_DIRS.install_dir
            }
        ]
        env_vars = MIMALLOC_ENV_VARS
    }
}

target "mimalloc_source" {
    contexts = {base = "target:source_getter"}
    dockerfile-inline = utils_fetch_git({
        host = "github.com"
        owner = "microsoft"
        repo = "mimalloc"
        rev = "dd7348066fe40e8bf372fa4e9538910a5e24a75f"
    })
}

target "mimalloc_install" {
    contexts = merge(
        {base = "target:cpp_builder"},
        utils_get_unpack_phase_contexts(MIMALLOC_BUILDER_ARGS)
    )
    dockerfile-inline = cmake_builder(MIMALLOC_BUILDER_ARGS)
}

target "mimalloc" {
    tags = ["connorbaker01/mimalloc:latest"]
    contexts = utils_get_unpack_phase_contexts(MIMALLOC_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(MIMALLOC_DISTRIBUTER_ARGS)
}
