FROM python:3.12-slim

# Installing uv (https://docs.astral.sh/uv/guides/integration/fastapi/#migrating-an-existing-fastapi-project)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container, into an app/ directory
COPY . /app

# Install app dependencies
WORKDIR /app
RUN uv sync --frozen --no-cache

# Run application
CMD ["/app/.env/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]