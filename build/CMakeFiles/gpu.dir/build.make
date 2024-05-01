# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/y/yt634/cs5220_final_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/y/yt634/cs5220_final_project/build

# Include any dependencies generated for this target.
include CMakeFiles/gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gpu.dir/flags.make

CMakeFiles/gpu.dir/main.cu.o: CMakeFiles/gpu.dir/flags.make
CMakeFiles/gpu.dir/main.cu.o: ../main.cu
CMakeFiles/gpu.dir/main.cu.o: CMakeFiles/gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/y/yt634/cs5220_final_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/gpu.dir/main.cu.o"
	/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/gpu.dir/main.cu.o -MF CMakeFiles/gpu.dir/main.cu.o.d -x cu -c /global/homes/y/yt634/cs5220_final_project/main.cu -o CMakeFiles/gpu.dir/main.cu.o

CMakeFiles/gpu.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gpu.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/gpu.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gpu.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target gpu
gpu_OBJECTS = \
"CMakeFiles/gpu.dir/main.cu.o"

# External object files for target gpu
gpu_EXTERNAL_OBJECTS =

gpu: CMakeFiles/gpu.dir/main.cu.o
gpu: CMakeFiles/gpu.dir/build.make
gpu: CMakeFiles/gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/y/yt634/cs5220_final_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpu.dir/build: gpu
.PHONY : CMakeFiles/gpu.dir/build

CMakeFiles/gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gpu.dir/clean

CMakeFiles/gpu.dir/depend:
	cd /global/homes/y/yt634/cs5220_final_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/y/yt634/cs5220_final_project /global/homes/y/yt634/cs5220_final_project /global/homes/y/yt634/cs5220_final_project/build /global/homes/y/yt634/cs5220_final_project/build /global/homes/y/yt634/cs5220_final_project/build/CMakeFiles/gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gpu.dir/depend

