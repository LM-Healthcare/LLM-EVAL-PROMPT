# prompteval: evaluation client (no GPU needed; the open model is served by the vllm service).
#
# The repository is bind-mounted at /app by docker-compose and PYTHONPATH points to /app/src,
# so code, configs and prompts are always the checked-out version: after `git pull` no rebuild
# is needed unless pyproject.toml dependencies change.

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[all,dev]"

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

ENTRYPOINT ["python", "-m", "prompteval.cli"]
CMD ["--help"]
