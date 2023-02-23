// Deps: utils, pip

variable "BSRT_DIRS" {
    default = {
        source_dir = "/opt/source/bsrt",
        build_dir = "/opt/build/bsrt",
        install_dir = "/opt/bsrt",
    }
}

variable "BSRT_ENV_VARS" {
    default = {
        PYTHONPATH = "${BSRT_DIRS.install_dir}:$PYTHONPATH",
    }
}

target "bsrt_spynet_checkpoint" {
    dockerfile-inline = <<-EOF
        FROM scratch
        ADD https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth .
    EOF
}

target "bsrt_alexnet_checkpoint" {
    dockerfile-inline = <<-EOF
        FROM scratch
        ADD https://download.pytorch.org/models/alexnet-owt-7be5be79.pth .
    EOF
}

target "bsrt_source" {
    dockerfile-inline = <<-EOF
        FROM scratch
        COPY ./ .
    EOF
}

variable "BSRT_PIP_INSTALL_ARGS" {
    default = {
        packages = {
            "${BSRT_DIRS.source_dir}[tune]" = null
        }
        flags = {
            target = BSRT_DIRS.install_dir
            verbose = true
        }
    }
}

variable "BSRT_BUILDER_ARGS" {
    default = {
        installers = [
            MIMALLOC_INSTALLER_ARGS,
            CPYTHON_INSTALLER_ARGS,
            COMMON_PYTHON_PACKAGES_INSTALLER_ARGS,
            TORCH_INSTALLER_ARGS,
            TORCHVISION_INSTALLER_ARGS,
            TRITON_INSTALLER_ARGS,
        ]

        unpack_phase = [
            {
                context = "bsrt_source"
                dest = BSRT_DIRS.source_dir
            },
        ]

        configure_phase = [
            "python3 -m sysconfig"
        ]
        
        install_phase = pip_install(BSRT_PIP_INSTALL_ARGS)
    }
}

variable "BSRT_DISTRIBUTER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "bsrt_install"
                source = BSRT_DIRS.install_dir
                dest = BSRT_DIRS.install_dir
            }
        ]
        env_vars = BSRT_ENV_VARS
    }
}


variable "BSRT_INSTALLER_ARGS" {
    default = {
        unpack_phase = [
            {
                context = "bsrt"
                source = BSRT_DIRS.install_dir
                dest = BSRT_DIRS.install_dir
            },
            {
                context = "bsrt_spynet_checkpoint"
                dest = "/root/.cache/torch/hub/checkpoints"
            },
            {
                context = "bsrt_alexnet_checkpoint"
                dest = "/root/.cache/torch/hub/checkpoints"
            },
        ]
        env_vars = BSRT_ENV_VARS
    }
}

target "bsrt_install" {
    contexts = merge(
        {base = "target:cuda_builder"},
        utils_get_unpack_phase_contexts(BSRT_BUILDER_ARGS)
    )
    dockerfile-inline = utils_builder(BSRT_BUILDER_ARGS)
}

target "bsrt" {
    tags = ["connorbaker01/bsrt:latest"]
    contexts = utils_get_unpack_phase_contexts(BSRT_DISTRIBUTER_ARGS)
    dockerfile-inline = utils_distributer(BSRT_DISTRIBUTER_ARGS)
}