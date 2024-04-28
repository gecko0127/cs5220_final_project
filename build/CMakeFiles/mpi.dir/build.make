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
include CMakeFiles/mpi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mpi.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi.dir/flags.make

CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o: CMakeFiles/mpi.dir/flags.make
CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o: ../mpi_openmp_main.cpp
CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o: CMakeFiles/mpi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/y/yt634/cs5220_final_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o -MF CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o.d -o CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o -c /global/homes/y/yt634/cs5220_final_project/mpi_openmp_main.cpp

CMakeFiles/mpi.dir/mpi_openmp_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpi.dir/mpi_openmp_main.cpp.i"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/homes/y/yt634/cs5220_final_project/mpi_openmp_main.cpp > CMakeFiles/mpi.dir/mpi_openmp_main.cpp.i

CMakeFiles/mpi.dir/mpi_openmp_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpi.dir/mpi_openmp_main.cpp.s"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/homes/y/yt634/cs5220_final_project/mpi_openmp_main.cpp -o CMakeFiles/mpi.dir/mpi_openmp_main.cpp.s

# Object files for target mpi
mpi_OBJECTS = \
"CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o"

# External object files for target mpi
mpi_EXTERNAL_OBJECTS =

mpi: CMakeFiles/mpi.dir/mpi_openmp_main.cpp.o
mpi: CMakeFiles/mpi.dir/build.make
mpi: /opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu_123_mpi_mp.so
mpi: /opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu_123_mp.so
mpi: /usr/lib64/gcc/x86_64-suse-linux/12/libgomp.so
mpi: /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/stubs/libcuda.so
mpi: /opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu_123_mpi.so
mpi: /opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib/libmpi_gnu_123.so
mpi: /opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_cuda.so
mpi: /opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu_123.so
mpi: /usr/lib64/libdl.so
mpi: /opt/cray/pe/dsmml/0.2.2/dsmml/lib/libdsmml.so
mpi: /opt/cray/xpmem/2.6.2-2.5_2.38__gd067c3f.shasta/lib64/libxpmem.so
mpi: CMakeFiles/mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/y/yt634/cs5220_final_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mpi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi.dir/build: mpi
.PHONY : CMakeFiles/mpi.dir/build

CMakeFiles/mpi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi.dir/clean

CMakeFiles/mpi.dir/depend:
	cd /global/homes/y/yt634/cs5220_final_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/y/yt634/cs5220_final_project /global/homes/y/yt634/cs5220_final_project /global/homes/y/yt634/cs5220_final_project/build /global/homes/y/yt634/cs5220_final_project/build /global/homes/y/yt634/cs5220_final_project/build/CMakeFiles/mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi.dir/depend

