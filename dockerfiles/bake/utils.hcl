// Deps: None

function "utils_unlines" {
    // Join a list of strings with newlines, ignoring empty strings
    // 
    // args: list[string] - list of strings to join
    params = [args]
    result = join("\n", compact(args))
}

function "utils_ands" {
    // Join a list of strings with "&&", ignoring empty strings
    // 
    // args: list[string] - list of strings to join
    params = [args]
    result = join(" && ", compact(args))
}

function "utils_spaces" {
    // Join a list of strings with spaces, ignoring empty strings
    // 
    // args: list[string] - list of strings to join
    params = [args]
    result = join(" ", compact(args))
}

function "utils_run" {
    // Run a list of commands in a single RUN statement, ignoring empty strings.
    // 
    // caches: optional[list[string]] - list of paths to provide to the run command as caches
    // commands: list[string] - list of commands to run
    params = [args]
    result = "RUN ${utils_ands(args)}"
}

function "utils_unpack_phase" {
    // Unpacks data from the source contexts into the destination context.
    // args: list[object]
    //  context: optional[string] - the source context. If specified, the source assumed to the 
    //      name of a target.
    //  source: optional[string] - the source directory. Defaults to ".".
    //  dest: optional[string] - the destination directory. Defaults to ".".
    params = [args]
    result = utils_unlines([
        for arg in args:
        utils_spaces([
            "COPY",
            try("--from=${arg.context}", ""),
            try(arg.source, "./"),
            try(arg.dest, ".")
        ])
    ])
}

function "utils_installer_unpack_phase_helper" {
    // Helper function for utils_unpack_phase which takes a list of installers and an unpack_phase 
    // and returns an updated unpack_phase.
    // 
    //  installers: optional[list[object]] - list of installers to use
    //  unpack_phase: optional[list[object]] - list of objects describing how to unpack data. See
    //      utils_unpack_phase.
    params = [args]
    result = distinct(concat(
        flatten([
            for installer in try(args.installers, []):
            try(installer.unpack_phase, [])
        ]),
        try(args.unpack_phase, [])
    ))
}

function "utils_export" {
    // Uses bash to export variables. To be used within a RUN command.
    // 
    // args: object - map of environment variables to set
    params = [args]
    result = utils_ands([for key, value in args: "export ${key}=\"${value}\""])
}

function "utils_envs" {
    params = [args]
    result = utils_unlines([
        for arg in args:
        utils_env(arg)
    ])
}

function "utils_env" {
    // Set environment variables in a single ENV statement, ignoring empty strings
    // 
    // args: object - map of environment variables to set
    params = [args]
    result = (
        length(values(args)) > 0
        ? "ENV ${utils_spaces([for key, value in args: "${key}=\"${value}\""])}"
        : ""
    )
}

function "utils_installer_env_vars_helper" {
    // Helper function for utils_env_vars which takes a list of installers and an env_vars and 
    // returns an updated env_vars.
    // 
    // args: list[object] - list of installers to use
    params = [args]
    // result = utils_unlines(distinct([
    //     for installer in args:
    //     utils_env(try(installer.env_vars, {}))
    // ]))
    result = utils_unlines([
        for installer in args:
        utils_env(try(installer.env_vars, {}))
    ])
}

function "utils_builder" {
    // Each of the configure, build, install, and fixup phases run in a separate RUN command. The 
    // pre and post-phases run in the same RUN command as their respective phases.
    // 
    // args: object
    //  image: optional[object] -- image to use as the base image. Defaults to {from = "base"}. 
    //      See utils_image.
    //
    //  unpack_phase: optional[list[object]] - list of objects describing how to unpack data. See 
    //      utils_unpack_phase. This argument is pass through distinct to avoid copying data from
    //      the same context multiple times.
    // 
    //  patch_phase: optional[list[string]] - list of bash commands to run in the patch phase
    // 
    //  pre_configure_phase: optional[list[string]] - list of bash commands to run in the pre-configure
    //      phase
    //  configure_phase: optional[list[string]] - list of bash commands to run in the configure phase
    //  post_configure_phase: optional[list[string]] - list of bash commands to run in the 
    //      post-configure phase
    //
    //  pre_build_phase: optional[list[string]] - list of bash commands to run in the pre-build phase
    //  build_phase: optional[list[string]] - list of bash commands to run in the build phase
    //  post_build_phase: optional[list[string]] - list of bash commands to run in the post-build phase
    //
    //  pre_install_phase: optional[list[string]] - list of bash commands to run in the pre-install
    //      phase
    //  install_phase: optional[list[string]] - list of bash commands to run in the install phase
    //  post_install_phase: optional[list[string]] - list of bash commands to run in the post-install
    //      phase
    //
    //  fixup_phase: optional[list[string]] - list of commands to run in the fixup phase

    params = [args]
    result = utils_unlines([
        utils_image(try(args.image, {from = "base"})),

        can(args.unpack_phase) || can(args.installers)
            ? utils_unpack_phase(utils_installer_unpack_phase_helper({
                installers = try(args.installers, []),
                unpack_phase = try(args.unpack_phase, [])
            }))
            : "",
        
        // Get the environment variables from the installers
        utils_installer_env_vars_helper(try(args.installers, [])),

        can(args.patch_phase)
            ? utils_run(args.patch_phase)
            : "",
        
        can(args.configure_phase)
            ? utils_run(concat(
                try(args.pre_configure_phase, []),
                args.configure_phase,
                try(args.post_configure_phase, []),
            ))
            : "",
        
        can(args.build_phase)
            ? utils_run(concat(
                try(args.pre_build_phase, []),
                args.build_phase,
                try(args.post_build_phase, []),
            ))
            : "",
        
        can(args.install_phase)
            ? utils_run(concat(
                try(args.pre_install_phase, []),
                args.install_phase,
                try(args.post_install_phase, []),
            ))
            : "",

        can(args.fixup_phase) ? utils_run(args.fixup_phase) : "",
    ])
}

function "utils_get_unpack_phase_contexts" {
    // Takes argument given to utils_unpack_phase and returns an object containing all of the 
    // contexts provided to the function.
    // 
    // args: object - See utils_unpack_phase.
    params = [args]
    result = {
        for arg in utils_installer_unpack_phase_helper({
            installers = try(args.installers, []),
            unpack_phase = try(args.unpack_phase, [])
        }):
        arg.context => "target:${arg.context}"
    }
}

// TODO: Does not handle case where are bash commands are in the install phase because we copy 
// into a scratch image, which has no shell.
function "utils_distributer" {
    // Takes the arguments provided to the utils_builder function and returns a list of commands to
    // create a dockerfile with the contents of the install phase.
    // 
    // args: object
    //  unpack_phase: list[string] - list of contexts to unpack
    //  env_vars: optional[object] - environment variables to set
    params = [args]
    result = utils_unlines([
        utils_image({
            from = "scratch"
            use_bash = false
        }),
        utils_unpack_phase(args.unpack_phase),
        utils_env(args.env_vars),
    ])
}

function "utils_dockerfile" {
    // Generate a Dockerfile from a list of strings.
    // 
    // Always starts with FROM base and sets the shell to bash.
    // 
    // Change the desired base image by passing a different image through the context.
    // 
    // args: list[string] - list of strings to join
    params = [args]
    result = utils_unlines(concat(
        [utils_image({from = "base"})],
        args
    ))
}

function "utils_fetch_git" {
    // Fetches a git repository.
    // args: object
    //  host: string - hostname of the repo
    //  owner: string - owner of the repo
    //  repo: string - repo name
    //  rev: string - revision to fetch
    params = [args]
    result = <<-EOF
        FROM base as fetched
        WORKDIR /root
        ${utils_run([
            "git clone https://${args.host}/${args.owner}/${args.repo} --recursive",
            "cd ${args.repo}",
            "git checkout ${args.rev} --recurse-submodules"
        ])}
        FROM scratch
        COPY --link --from=fetched /root/${args.repo} .
    EOF
}

function "utils_fetch_tarball" {
    // Fetches and unpacks a tarball.
    //  url: string - url of the tarball
    params = [args]
    result = <<-EOF
        FROM base as fetched
        WORKDIR /root
        ${utils_run([
            "curl -L ${args.url} -o tarball",
            "mkdir extracted",
            "tar -xf tarball -C extracted --strip-components=1",
            "rm tarball"
        ])}
        FROM scratch
        COPY --link --from=fetched /root/extracted .
    EOF
}

function "utils_image" {
    // From statement. Sets the shell to bash.
    // 
    // args: object
    //  from: string - image to use
    //  sha256: optional[string] - sha256 of the image; if not specified, the image is not pinned
    //  use_bash: optional[bool] - if true, sets the shell to bash, otherwise uses the default 
    //      shell. Default is true.
    params = [args]
    result = utils_unlines([
        "# syntax=docker/dockerfile-upstream:master-labs",

        "FROM ${try("${args.from}@sha256:${args.sha256}", args.from)}",
        
        try(args.use_bash, true) ? "SHELL [\"/bin/bash\", \"-c\"]" : "",
    ])
}

function "utils_format_flags" {
    // Format a map of flags into a string. Flags with a value of false are ignored. Flags with a 
    // value of true are formatted as a single flag. Flags with a value other than true or false 
    // are formatted as a flag with a value.
    // 
    // args: object
    //  flags: object - map of flags to values
    //  key_prefix: optional[string] - prefix to add to each flag's key
    params = [args]
    result = utils_spaces([
        for key, value in args.flags: 
        // Try to cast value to a bool. If it's not the literals "true" or "false", then it's a
        // string and we should format it as a flag with a value.
        try(
            value ? "${try(args.key_prefix, "")}${key}" : "",
            "${try(args.key_prefix, "")}${key}=${value}"
        )
    ])
}