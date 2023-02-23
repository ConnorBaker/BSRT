// Deps: utils
function "_pip_flags" {
    // Return a list of pip flags.
    //
    // args: object
    //  pre: optional[bool] - whether to install pre-release versions
    //  extra_index_urls: optional[list[string]] - list of extra index URLs to use
    //  no_cache_dir: optional[bool] - whether to disable the cache. Defaults to true.
    //  no_deps: optional[bool] - whether to skip installing dependencies
    //  no_build_isolation: optional[bool] - whether to disable build isolation. Defaults to true.
    //  target: optional[string] - directory to install packages to
    //  verbose: optional[bool] - whether to enable verbose output
    //  editable: optional[bool] - whether to install packages in editable mode
    params = [args]
    result = utils_spaces([
        try(args.pre, false) ? "--pre" : "",
        can(args.extra_index_urls)
                ? utils_spaces([
                    for url in args.extra_index_urls: 
                    "--extra-index-url=${url}"
                ])
                : "",
        try(args.no_cache_dir, true) ? "--no-cache-dir" : "",
        try(args.no_deps, false) ? "--no-deps" : "",
        try(args.no_build_isolation, true) ? "--no-build-isolation" : "",
        can(args.target) ? "--target=${args.target}" : "",
        try(args.verbose, false) ? "--verbose" : "",
        // NOTE: Editable must be last so the local package immediately follows it.
        try(args.editable, false) ? "--editable" : "",
    ])
}

function "pip_install" {
    // Return a list of commands to install packages with pip.
    //
    // args: object
    //  pre_install: optional[list[string]] - list of commands to run before installing packages
    //  post_install: optional[list[string]] - list of commands to run after installing packages
    //  packages: object - map of package names to versions
    //  flags: optional[object] - map of flags to pass to pip. See _pip_flags.
    params = [args]
    result = concat(
        try(args.pre_install, []),
        [
            utils_spaces([
                "python3 -m pip install",
                _pip_flags(try(args.flags, {})),
                utils_spaces([
                    for name, version in args.packages:
                    // If version is null, just use name.
                    // Allows us to install local packages.
                    version != null
                        ? "${name}==${version}"
                        : "${name}"
                ]),
            ])
        ],
        try(args.post_install, []),
    )
}