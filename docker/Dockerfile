# syntax=docker/dockerfile:1
# lm-mxl-mixer: CPU vision mixer for MXL.
#   docker build -f docker/Dockerfile -t lm-mxl-mixer:dev .
# MXL pin: dmf-mxl/mxl release/v1.1 218ddaa0a08c12ffe75fc475ae65aa3d9eef16d7, fabrics off
# (the mxl-stack functions' pin).

FROM ubuntu:24.04 AS mxl
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates zip unzip tar pkg-config uuid-dev python3 \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/build-mxl.sh /usr/local/bin/build-mxl.sh
RUN VCPKG_ROOT=/opt/vcpkg MXL_BUILD_DIR=/tmp/mxl /usr/local/bin/build-mxl.sh /opt/mxl \
    && rm -rf /tmp/mxl /opt/vcpkg/buildtrees /opt/vcpkg/downloads

FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git pkg-config clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY web ./web
COPY examples ./examples
RUN --mount=type=cache,target=/opt/cargo/registry --mount=type=cache,target=/opt/cargo/git \
    cargo build --release --locked --bins \
    && install -m 0755 target/release/lm-mxl-mixer target/release/mxl-pattern target/release/mxl-grab /usr/local/bin/

FROM ubuntu:24.04 AS runtime
ARG MXL_REF=218ddaa0a08c12ffe75fc475ae65aa3d9eef16d7
ARG VERSION=dev
# ffmpeg (x264) encodes the browser preview.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libstdc++6 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /config && chown 1000:1000 /config
COPY --from=mxl /opt/mxl/lib/libmxl.so* /usr/local/lib/
COPY --from=build /usr/local/bin/lm-mxl-mixer /usr/local/bin/mxl-pattern /usr/local/bin/mxl-grab /usr/local/bin/
RUN ldconfig && ldd /usr/local/lib/libmxl.so.1 | (! grep -q "not found") \
    && ldd /usr/local/bin/lm-mxl-mixer | (! grep -q "not found")
USER 1000:1000
ENV MIXER_STATE_DIR=/config
# Web/API/metrics, NMOS.
EXPOSE 8180 3332
LABEL org.opencontainers.image.title="lm-mxl-mixer" \
      org.opencontainers.image.description="CPU vision mixer for MXL (Livemotion)" \
      org.opencontainers.image.version="${VERSION}" \
      io.dmf.mxl.revision="${MXL_REF}"
ENTRYPOINT ["/usr/local/bin/lm-mxl-mixer"]
