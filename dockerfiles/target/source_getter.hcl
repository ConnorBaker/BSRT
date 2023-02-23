// Deps: utils, apt

target "source_getter" {
    tags = ["connorbaker01/source_getter:latest"]
    dockerfile-inline = utils_unlines([
        utils_image({
            from = "docker.io/bitnami/git:2.39.1-debian-11-r6"
            sha256 = "fc523885cd783b59b2a03eb033b8aa67d9425f6f2ceb16124b43834577945e22"
        }),
        utils_env({
            DEBIAN_FRONTEND = "noninteractive"
        }),
        utils_run(concat(
            [
                "git config --global advice.detachedHead false",
                "git config --global init.defaultBranch main",
                "git config --global submodule.recurse true",
                "git config --global user.email source_getter@source_getter",
                "git config --global user.name source_getter"
            ],
            apt_install(["xz-utils"])
        ))
    ])
}
