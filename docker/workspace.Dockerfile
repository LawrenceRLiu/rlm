FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        matplotlib \
        sympy \
        beautifulsoup4 \
        lxml \
        pyyaml \
        pillow \
        pypdf \
        openpyxl \
        regex \
        requests \
        httpx \
        flask

COPY workspace_image/rlm_workspace/ /opt/rlm_workspace/rlm_workspace/
ENV PYTHONPATH=/opt/rlm_workspace
ENV RLM_BROKER_PORT=8080

WORKDIR /workspace

EXPOSE 8080

CMD ["python", "-m", "rlm_workspace.broker"]
