# WebGPU-Template-Project

## Quick Start
### What you need first?
CUDA on WSL2 requires your Windows from Dev Channel (build version 20145 or later), current Windows 11 (version 21H2) has build 22000. As of time of writing, Microsoft just announced that next version is going to be on Beta channel, so stability-wise it should be fine.

todo: packages

### How to use this?
1) Clone this repo using your favorite method.
2) You need to pull the `libwb` submodule. Use `init` and `recursive` to make sure you fetch all new submodules.
    ```
    git submodule update --init --recursive
    ```
3) Create a build folder, we are going to use `build` here. Generated build files and configs will store here.
    ```
    mkdir build
    cd build
    ```
4) Let `cmake` do its magic and compile!
    ```
    cmake ..
    make
    ```
   This should first compile `libwb` and then compile MP0 with ease.
   Output binary `mp0` is copied to `bin` (same level as your `build` folder).
5) Run your MP0.
   ```
   ./mp0
   ```

## Tested Environments
| OS | CMake | CXX | CUDA |
| :-: | :-: | :-: | :-: |
| WSL2 5.10.60.1 | 3.16.3 | gcc 9.3.0 | 11.2.152 |
