# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build

# Include any dependencies generated for this target.
include CMakeFiles/sfm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sfm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sfm.dir/flags.make

CMakeFiles/sfm.dir/src/io_utils.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/io_utils.cpp.o: ../src/io_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sfm.dir/src/io_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/src/io_utils.cpp.o -c /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/io_utils.cpp

CMakeFiles/sfm.dir/src/io_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/io_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/io_utils.cpp > CMakeFiles/sfm.dir/src/io_utils.cpp.i

CMakeFiles/sfm.dir/src/io_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/io_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/io_utils.cpp -o CMakeFiles/sfm.dir/src/io_utils.cpp.s

CMakeFiles/sfm.dir/src/features_matcher.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/features_matcher.cpp.o: ../src/features_matcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sfm.dir/src/features_matcher.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/src/features_matcher.cpp.o -c /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/features_matcher.cpp

CMakeFiles/sfm.dir/src/features_matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/features_matcher.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/features_matcher.cpp > CMakeFiles/sfm.dir/src/features_matcher.cpp.i

CMakeFiles/sfm.dir/src/features_matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/features_matcher.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/features_matcher.cpp -o CMakeFiles/sfm.dir/src/features_matcher.cpp.s

CMakeFiles/sfm.dir/src/basic_sfm.cpp.o: CMakeFiles/sfm.dir/flags.make
CMakeFiles/sfm.dir/src/basic_sfm.cpp.o: ../src/basic_sfm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sfm.dir/src/basic_sfm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sfm.dir/src/basic_sfm.cpp.o -c /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/basic_sfm.cpp

CMakeFiles/sfm.dir/src/basic_sfm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm.dir/src/basic_sfm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/basic_sfm.cpp > CMakeFiles/sfm.dir/src/basic_sfm.cpp.i

CMakeFiles/sfm.dir/src/basic_sfm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm.dir/src/basic_sfm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/src/basic_sfm.cpp -o CMakeFiles/sfm.dir/src/basic_sfm.cpp.s

# Object files for target sfm
sfm_OBJECTS = \
"CMakeFiles/sfm.dir/src/io_utils.cpp.o" \
"CMakeFiles/sfm.dir/src/features_matcher.cpp.o" \
"CMakeFiles/sfm.dir/src/basic_sfm.cpp.o"

# External object files for target sfm
sfm_EXTERNAL_OBJECTS =

libsfm.a: CMakeFiles/sfm.dir/src/io_utils.cpp.o
libsfm.a: CMakeFiles/sfm.dir/src/features_matcher.cpp.o
libsfm.a: CMakeFiles/sfm.dir/src/basic_sfm.cpp.o
libsfm.a: CMakeFiles/sfm.dir/build.make
libsfm.a: CMakeFiles/sfm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libsfm.a"
	$(CMAKE_COMMAND) -P CMakeFiles/sfm.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sfm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sfm.dir/build: libsfm.a

.PHONY : CMakeFiles/sfm.dir/build

CMakeFiles/sfm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sfm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sfm.dir/clean

CMakeFiles/sfm.dir/depend:
	cd /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build /home/somayeh/CLionProjects/3Dprocessing/3D_structure_from_motion/sfm/build/CMakeFiles/sfm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sfm.dir/depend

