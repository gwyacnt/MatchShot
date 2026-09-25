# Supported target: Linux x86_64 with an AMD GPU supported by ROCm and Vulkan.
FROM rocm/dev-ubuntu-24.04:7.2.1@sha256:fe29799baf7b8db986a707110c15579dc0fb6e39d48973fbb82dfc33a0c7e179
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PHOTO_SELECTOR_STATE=/state
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates libgl1 libglib2.0-0 \
    libgomp1 libssl3t64 libstdc++6 migraphx rocminfo \
    mesa-vulkan-drivers libvulkan1 && rm -rf /var/lib/apt/lists/*
COPY .devcontainer/requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps --require-hashes -r /tmp/requirements.lock
ARG USER_UID=1000
ARG USER_GID=1000
RUN if getent passwd ubuntu >/dev/null; then userdel -r ubuntu; fi \
    && (getent group ${USER_GID} >/dev/null || groupadd --gid ${USER_GID} vscode) \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash vscode
ENV PATH=/home/vscode/.local/bin:/opt/rocm/bin:${PATH}
WORKDIR /workspaces/photo_selector
USER vscode
CMD ["sleep", "infinity"]
