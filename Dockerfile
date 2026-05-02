# Base image with Python 3.11
FROM python:3.11-slim

# Set the primary working directory
WORKDIR /app

# Install system dependencies required by the libsumo C++ extension.
# Placed early in the file to leverage Docker layer caching.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libatomic1 \
    libfontconfig1 \
    sumo \
    sumo-tools \
    && rm -rf /var/lib/apt/lists/*

# Set the SUMO_HOME environment variable to point to the installed SUMO directory.
ENV SUMO_HOME=/usr/share/sumo

# Copy and install Python dependencies separately to cache the pip install step.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container.
COPY . /app

# Ensure the entrypoint script is executable inside the container.
RUN chmod +x /app/entrypoint.sh

# Use the entrypoint script as the container's entrypoint.
# By default it runs the full pipeline (setup + IRRG).
# Override with: docker run tls_optimization <command>
#   setup       - generation + netconvert only
#   run         - IRRG only (assumes setup was done before)
#   all         - full pipeline (default)
#   <arbitrary> - pass-through (e.g. bash, python ...)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["all"]