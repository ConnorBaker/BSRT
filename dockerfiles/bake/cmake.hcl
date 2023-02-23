// Deps: utils, llvm, linker

function "cmake_format_flags" {
    // Format a map of CMake flags into a string. Boolean values are changed to "OFF" and "ON" 
    // so they are not ignored by utils_format_flags. Other values are passed through.
    //
    // args: object
    //  flags: object - flags to format
    params = [args]
    result = utils_format_flags({
        flags = {
            for key, value in args.flags:
            // utils_format_flags ignores flags with a value of false. We want to pass "OFF" and "ON"
            // instead.
            // Try to convert the value to a boolean. If it fails, pass the value through.
            key => "\"${try(value ? "ON" : "OFF", value)}\""
        }
        key_prefix = "-D"
    })
}

function "cmake_flags" {
    // Format a map of CMake flags into a string.
    // 
    // args: object
    //  cmake: object   
    //      build_shared_libs: optional[bool] - build shared libs, true by default
    //      build_type: optional[string] - build type, Release by default
    //      verbose: optional[bool] - verbose, false by default
    //      c_standard: optional[int] - C standard, 17 by default
    //      c_standard_required: optional[bool] - C standard required, true by default
    //      c_extensions: optional[bool] - C extensions, false by default
    //      cxx_standard: optional[int] - C++ standard, 17 by default
    //      cxx_standard_required: optional[bool] - C++ standard required, true by default
    //      cxx_extensions: optional[bool] - C++ extensions, false by default
    //      include_default_flags: optional[bool] - use all of the above default flags, true by 
    //          default
    //      extra_flags: optional[object] - additional flags
    //  compiler: object - llvm_compiler_flags() args object
    //  linker: object - linker_flags() args object
    params = [args]
    result = cmake_format_flags({
        flags = merge(
            try(args.cmake.include_default_flags, true)
                ? {
                    // Base CMake settings
                    BUILD_SHARED_LIBS = try(args.cmake.build_shared_libs, true)
                    CMAKE_BUILD_TYPE = try(args.cmake.build_type, "Release")
                    CMAKE_VERBOSE_MAKEFILE = try(args.cmake.verbose, false)

                    // Set C standard and flags
                    CMAKE_C_STANDARD = try(args.cmake.c_standard, 17)
                    CMAKE_C_STANDARD_REQUIRED = try(args.cmake.c_standard_required, true)
                    CMAKE_C_EXTENSIONS = try(args.cmake.c_extensions, false)
                    // Augment compiler flags with custom flags if they exist.
                    // Otherwise, use the compiler flags from the compiler object.
                    CMAKE_C_FLAGS = "${llvm_compiler_flags(try(args.compiler, {}))}"

                    // Set C++ standard and flags
                    CMAKE_CXX_STANDARD = try(args.cmake.cxx_standard, 17)
                    CMAKE_CXX_STANDARD_REQUIRED = try(args.cmake.cxx_standard_required, true)
                    CMAKE_CXX_EXTENSIONS = try(args.cmake.cxx_extensions, false)
                    // Augment compiler flags with custom flags if they exist.
                    // Otherwise, use the default compiler flags.
                    CMAKE_CXX_FLAGS = "${llvm_compiler_flags(try(args.compiler, {}))}"

                    // Set linker flags
                    // Augment linker flags with custom flags if they exist.
                    // Otherwise, use the default linker flags.
                    CMAKE_EXE_LINKER_FLAGS = "${linker_flags(try(args.linker, {}))} ${llvm_compiler_flags(try(args.compiler, {}))}"
                    CMAKE_MODULE_LINKER_FLAGS = "${linker_flags(try(args.linker, {}))} ${llvm_compiler_flags(try(args.compiler, {}))}"
                    CMAKE_SHARED_LINKER_FLAGS = "${linker_flags(try(args.linker, {}))} ${llvm_compiler_flags(try(args.compiler, {}))}"

                    // LTO policies
                    CMAKE_POLICY_DEFAULT_CMP0069 = "NEW"
                    CMAKE_POLICY_DEFAULT_CMP0105 = "NEW"
                    CMAKE_POLICY_DEFAULT_CMP0138 = "NEW"

                    // LTO
                    CMAKE_INTERPROCEDURAL_OPTIMIZATION = try(args.cmake.interprocedural_optimization, true)
                    CMAKE_POSITION_INDEPENDENT_CODE = try(args.cmake.position_independent_code, true)
                }
                : {},
            try(args.cmake.extra_flags, {})
        )
    })
}


function "cmake_configure_phase" {
    // Return a shell command to run CMake to configure a project.
    //
    // args: object
    //  source_dir: string - source directory
    //  build_dir: string - build directory
    //  generator: optional[string] - generator, Ninja by default
    //  flags: optional[object] - cmake_flags() args object
    params = [args]
    result = utils_spaces([
        "cmake -S ${args.source_dir} -B ${args.build_dir} -G ${try(args.generator, "Ninja")}",
        cmake_flags(try(args.flags, {}))
    ])
}

function "cmake_build_phase" {
    // Return a shell command to run CMake to build a project.
    //
    // args: object
    //  build_dir: string - build directory
    //  target: optional[string] - target, all by default
    params = [args]
    result = "cmake --build ${args.build_dir} --target ${try(args.target, "all")}"
}

function "cmake_install_phase" {
    // Return a shell command to run CMake to install a project.
    //
    // args: object
    //  build_dir: string - build directory
    //  install_dir: string - install prefix
    params = [args]
    result = "cmake --install ${args.build_dir} --prefix ${args.install_dir}"
}

function "cmake_builder" {
    // Wraps a utils_builder to add CMake support.
    // 
    // Accepts the same flags as utils_builder(), but changes the following:
    // 
    // args: object
    //  configure_phase: optional[object] - object describing how to configure the project. See
    //      cmake_configure_phase.
    //  build_phase: optional[object] - object describing how to build the project. See
    //      cmake_build_phase.
    //  install_phase: optional[object] - object describing how to install the project. See
    //      cmake_install_phase.
    params = [args]
    result = utils_builder(merge(
        args,
        can(args.configure_phase) ? {
            configure_phase = [cmake_configure_phase(args.configure_phase)]
        } : {},
        can(args.build_phase) ? {
            build_phase = [cmake_build_phase(args.build_phase)]
        } : {},
        can(args.install_phase) ? {
            install_phase = [cmake_install_phase(args.install_phase)]
        } : {}
    ))
}
