FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first so code edits don't reinstall them. Same lock file as dev = same versions.
# shortcut: trainer and API share one dependency set, so the image carries unused packages
# (fastapi etc.); split dependency groups if image size starts to matter.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN useradd --create-home --uid 1000 app && mkdir -p data && chown app data
USER app

CMD ["sh", "-c", "python -m src.data.fetch && python -m src.training.train"]
