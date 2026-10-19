FROM node:22-bookworm-slim

ARG CLAUDE_CODE_VERSION=2.1.231

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git openssh-client procps python3 ripgrep util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version

RUN useradd --uid 10250 --gid 0 --create-home --shell /bin/bash agent \
    && mkdir -p /workspace \
    && chown agent:root /workspace

USER agent
ENV HOME=/home/agent
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/claude"]
