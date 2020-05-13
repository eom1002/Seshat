
if(NOT "/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitinfo.txt" IS_NEWER_THAN "/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/eom1002/Seshat/build/_deps/xtensor-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/eom1002/Seshat/build/_deps/xtensor-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/xtensor-stack/xtensor.git" "xtensor-src"
    WORKING_DIRECTORY "/home/eom1002/Seshat/build/_deps"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/xtensor-stack/xtensor.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout master --
  WORKING_DIRECTORY "/home/eom1002/Seshat/build/_deps/xtensor-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/eom1002/Seshat/build/_deps/xtensor-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/eom1002/Seshat/build/_deps/xtensor-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitinfo.txt"
    "/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/eom1002/Seshat/build/_deps/xtensor-subbuild/xtensor-populate-prefix/src/xtensor-populate-stamp/xtensor-populate-gitclone-lastrun.txt'")
endif()

