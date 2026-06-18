FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# Install essential build tools and libraries for Mitsuba 0.6
# This includes:
# - build-essential: For compilers (g++, make, etc.)
# - python2.7-dev: Python 2.7 development headers and libraries
# - scons: The build system used by Mitsuba
# - libpng-dev, libjpeg-dev, libopenexr-dev, zlib1g-dev, libilmbase-dev, : Image format libraries
# - libboost-all-dev: General Boost C++ libraries
# - libxerces-c-dev: XML parser often used by Mitsuba
# - libglewmx-dev, libglu1-mesa-dev, mesa-common-dev: OpenGL and GLUT libraries
# - libxxf86vm-dev: X11 video mode extension library
# - libpcrecpp0v5: Perl-compatible regular expressions library
# - libeigen3-dev: C++ template library for linear algebra
# - libfftw3-dev: Fast Fourier Transform library
# - git: Useful if you need to clone repositories within the container

RUN apt-get update && apt-get install -y \
    build-essential \
    python2.7 \
    python2.7-dev \
    python3 \
    python3-pip \
    scons \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    zlib1g-dev \
    libilmbase-dev \
    libboost-all-dev \
    libxerces-c-dev \
    libglewmx-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libxxf86vm-dev \
    libpcrecpp0v5 \
    libeigen3-dev \
    libfftw3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the Mitsuba source code inside the container
WORKDIR /home

# Append the source command to the root user's .bashrc file
# This ensures it runs every time you open a new shell via docker exec
RUN echo "source /home/mitsuba/setpath.sh" >> /root/.bashrc
