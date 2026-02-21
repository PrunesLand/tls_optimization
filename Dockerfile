# Base image with Python 3.9
FROM python:3.9-slim

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
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies separately to cache the pip install step.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container.
COPY . /app

# Default command to execute the main application.
CMD ["python", "main.py"]