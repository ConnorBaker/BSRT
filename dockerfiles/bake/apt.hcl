// Deps: utils, apt

// TODO: Wipe the apt cache / extra files to reduce image size

variable "apt_RUN_CACHE" {
    default = "--mount=type=cache,target=/var/cache/apt"
}

function "apt_install" {
    // Returns a list of commands which install packages with apt.
    // 
    // args: list[string] - list of packages to install
    params = [args]
    result = [
        "apt update",
        "apt install -y --no-install-recommends ${utils_spaces(args)}",
        "rm -rf /var/lib/apt/lists/*"
    ]
}

function "apt_upgrade" {
    // Returns a list of commands which upgrade packages with apt.
    // 
    // args: object
    //  unhold: optional[list[string]] - list of packages to unhold before upgrading
    //  hold: optional[list[string]] - list of packages to hold before upgrading
    //  install: optional[list[string]] - list of packages to install before upgrading
    //  packages: optional[list[string]] - list of packages to upgrade. If not specified, all 
    //      packages will be upgraded.
    params = [args]
    result = compact([
        "apt update",
        can(args.unhold)
            ? "apt-mark unhold ${utils_spaces(args.unhold)}"
            : "",
        can(args.hold)
            ? "apt-mark hold ${utils_spaces(args.hold)}"
            : "",
        can(args.install)
            ? "apt install -y --no-install-recommends ${utils_spaces(args.install)}"
            : "",
        "apt upgrade -y --no-install-recommends ${try(utils_spaces(args.packages), "")}",
        "rm -rf /var/lib/apt/lists/*"
    ])
}

function "apt_add_repo" {
    // Returns a list of commands which add a repo to apt.
    // 
    // args: object
    //  repo_name: string - name of the repo
    //  repo_url: string - URL of the repo
    //  release: string - release name
    //  pub_key_url: string - URL of the public key
    //  keyring_name: string - name of the keyring
    params = [args]
    result = [
        utils_spaces([
            "curl -sL ${args.pub_key_url}",
            "|",
            "gpg --dearmor -o /usr/share/keyrings/${args.keyring_name}.gpg"
        ]),
        utils_spaces([
            "echo",
            "\"deb [signed-by=/usr/share/keyrings/${args.keyring_name}.gpg] ${args.repo_url} ${args.release} main\"",
            "|",
            "tee /etc/apt/sources.list.d/${args.repo_name}.list > /dev/null"
        ])
    ]
}
