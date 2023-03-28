# Iris
This is a simple OpenGL application whose main purpose is learning Graphics in general and more modern C++. A TODO list will be added and updated as I progress through this journey.

You should be able to compile & run this under *most* Windows or Linux machines with the following requirements:
- `OpenGL>=4.6;`
- `clang>=9;`
- `gcc>=8;`

The keybindings are as follows:
- `ESC`: Exits the application;
- `F1`: Recaptures the frustum;
- `W,A,S,D`: FPS camera movement;
- `F`: Displays the AABB boxes of the objects;
- `Right Mouse Button`: Enables the camera to be moved around;
- `Left Mouse Button`: Pick objects in 3D space;

## Building
To build this project you will need [CMake](https://cmake.org/) and a C++ compiler. Futhermore you will need to install
[Python](https://www.python.org/) and [Pip](https://pypi.org/project/pip/) to install the Python dependencies (`jinja2`).

Installing `jinja2` requires `pip install jinja2` in your terminal;
Compiling the project requires the following commands:
- `git clone --recursive [This Repo]`
- `cd [This Repo]`
- `mkdir build && cd build`
- `cmake .. -G [Your Favourite Generator]`
- `cmake --build . --target [target] -j[threads]`

Where `[target]` is any of:
- MousePicking
- Framebuffers

## Assets
To actually run the targets present in this you will need **assets**, reference the [models](models) folder.
