# libbioneuron

Weight solver for Two-Compartment LIF neurons using the OSQP library

## Compiling

This library has no run-time dependencies. To build, you'll need a C++11 compliant C++ compiler and a C89 compliant C compiler.

This library uses the `meson` build system, which uses `ninja`. To install both, run
```sh
sudo dnf install ninja-build # Fedora, RedHat, CentOS
sudo apt-get install ninja-build # Ubuntu, Debian
sudo pip3 install meson
```

Then, to compile the library, simply run
```sh
git clone https://github.com/astoeckel/libbioneuron
cd libbioneuron; mkdir build; cd build
meson .. -Dbuildtype=release
ninja
```

