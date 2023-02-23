// Deps: utils

function "llvm_compiler_flags" {
    // Format a map of compiler flags into a string. Flags with a value of false are ignored. Flags
    // with a value of true are formatted as a single flag. Flags with a value other than true or
    // false are formatted as a flag with a value.
    //
    // args: object
    //  optimize: optional[bool] - optimize with O3, true by default
    //  march: optional[string] - architecture to target, native by default
    //  pipe: optional[bool] - use pipes, true by default
    //  align_functions: optional[int] - align functions, 32 by default
    //  no_semantic_interposition: optional[bool] - disable semantic interposition, true by default
    //  function_sections: optional[bool] - put functions in their own sections, true by default
    //  data_sections: optional[bool] - put data in their own sections, true by default
    //  lto: optional[string] - link time optimization, thin by default
    //  split_lto_unit: optional[bool] - split LTO units, true by default. Requires lto.
    //  whole_program_vtables: optional[bool] - whole program vtables, true by default. Requires 
    //      lto.
    //  split_machine_functions: optional[bool] - split machine functions, true by default.
    //  slp_vectorize: optional[bool] - slp vectorize, true by default
    //  force_emit_vtables: optional[bool] - force emit vtables, true by default
    //  strict_vtable_pointers: optional[bool] - strict vtable pointers, true by default
    //  no_plt: optional[bool] - no plt, true by default
    //  no_common: optional[bool] - no common, true by default
    //  include_default_flags: optional[bool] - use all of the above default flags, true by default
    //  extra_flags: optional[object] - additional flags
    params = [args]
    result = utils_format_flags({
        flags = merge(
            try(args.include_default_flags, true)
                ? {
                        "-O3" = try(args.optimize, true)
                        "-march" = try(args.march, "native")
                        "--pipe" = try(args.pipe, true)
                        "-falign-functions" = try(args.align_functions, 32)
                        "-fno-semantic-interposition" = try(args.no_semantic_interposition, true)
                        "-ffunction-sections" = try(args.function_sections, true)
                        "-fdata-sections" = try(args.data_sections, true)
                        "-flto" = try(args.lto, "thin")
                        "-fsplit-lto-unit" = try(args.split_lto_unit, true)
                        "-fwhole-program-vtables" = try(args.whole_program_vtables, true)
                        "-fsplit-machine-functions" = try(args.split_machine_functions, true)
                        "-fslp-vectorize" = try(args.slp_vectorize, true)
                        "-fforce-emit-vtables" = try(args.force_emit_vtables, true)
                        "-fstrict-vtable-pointers" = try(args.strict_vtable_pointers, true)
                        "-fno-plt" = try(args.no_plt, true)
                        "-fno-common" = try(args.no_common, true)
                    }
                : {},
            try(args.extra_flags, {})
        )
    })
}