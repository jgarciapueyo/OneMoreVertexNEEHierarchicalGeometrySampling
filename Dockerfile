# Build and run environment for One-more-vertex NEE with Hierarchical Geometry
# Sampling (Mitsuba 0.6). The repository is mounted at /home/mitsuba (see
# docker-compose.yml) and compiled inside the container with
# scripts/build_mitsuba.sh; the PPG baseline is built into the image.
FROM ubuntu:20.04@sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Mitsuba 0.6 build dependencies (SCons runs on Python 2.7), Python 3 for the
# experiment scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python2.7 \
    python2.7-dev \
    python3 \
    python3-pip \
    scons \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libilmbase-dev \
    zlib1g-dev \
    libboost-all-dev \
    libxerces-c-dev \
    libglewmx-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libxxf86vm-dev \
    libpcrecpp0v5 \
    libeigen3-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies of scripts/ and of the scene generators
COPY scripts/experiments/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Practical Path Guiding baseline (Mueller et al. 2017) for Figs. 9 and 10,
# pinned and patched (binary renamed to mitsuba-ppg)
ARG PPG_COMMIT=effbd1934a87bd7c471bb6981eea758fc4b72854
COPY patches/ppg.patch /tmp/ppg.patch
RUN git clone https://github.com/Tom94/practical-path-guiding.git /opt/ppg && \
    cd /opt/ppg && git checkout ${PPG_COMMIT} && git apply /tmp/ppg.patch && \
    cd mitsuba && cp build/config-linux-gcc.py config.py && \
    python2.7 $(which scons) -j"$(nproc)" && \
    rm -rf build/release .sconf_temp .sconsign.dblite

# Unprivileged user with the host's UID/GID, so that files written to the
# mounted repository (build/, dist/, results) belong to the host user
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g ${GID} mitsuba && \
    useradd -o -m -d /home/user -u ${UID} -g ${GID} -s /bin/bash mitsuba && \
    git config --system --add safe.directory /home/mitsuba

# What setpath.sh sets for the build in the mounted repository, so that
# mitsuba is available in every shell
ENV MITSUBA_DIR=/home/mitsuba
ENV PATH=/home/mitsuba/dist:${PATH}
ENV LD_LIBRARY_PATH=/home/mitsuba/dist

# Tells scripts/experiments/run_experiments.py to call Mitsuba directly
ENV ONEEHGS_IN_CONTAINER=1

USER mitsuba
WORKDIR /home/mitsuba
CMD ["bash"]
