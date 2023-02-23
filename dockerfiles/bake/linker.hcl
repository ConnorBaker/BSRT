// Deps: utils

function "linker_flags" {
    // Format a map of linker flags into a string. Flags with a value of false are ignored. Flags
    // with a value of true are formatted as a single flag. Flags with a value other than true or
    // false are formatted as a flag with a value.
    //
    // args: object
    //  use_ld: optional[string] - linker to use, whatever $LD is set to by default
    //  optimize: optional[bool] - optimize with O2, true by default
    //  z_now: optional[bool] - use z now, true by default
    //  z_nodlopen: optional[bool] - use z nodlopen, false by default
    //  sort_common: optional[bool] - sort common, true by default
    //  gc_sections: optional[bool] - gc sections, true by default
    //  hash_style: optional[string] - hash style, gnu by default
    //  include_default_flags: optional[bool] - use all of the above default flags, true by default
    //  extra_flags: optional[object] - additional flags
    params = [args]
    result = utils_format_flags({
        flags = merge(
            try(args.include_default_flags, true)
                ? {
                    "-fuse-ld" = try(args.use_ld, "$LD")
                    "-O2" = try(args.optimize, true)
                    "-znow" = try(args.z_now, true)
                    // NOTE: New in mold 1.10
                    //  [x86-64][s390x] mold now optimizes thread-local variable accesses in 
                    //  shared libraries if the library is linked with -z nodlopen. If your shared 
                    //  library is not intended to be used via dlopen(2) and your library 
                    //  frequently accesses thread-local variables, you might want to pass that 
                    //  option when linking your library. (25d02bb, f32ce33)
                    "-znodlopen" = try(args.z_nodlopen, false)
                    "--sort-common" = try(args.sort_common, true)
                    "--gc-sections" = try(args.gc_sections, true)
                    "--hash-style" = try(args.hash_style, "gnu")
                }
                : {},
            try(args.extra_flags, {})
        )
        key_prefix = "-Wl,"
    })
}