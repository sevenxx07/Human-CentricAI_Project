FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    build-essential \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    python3-dev \
    cmake \
    git \
    libtbb-dev \
    ninja-build \
    pkg-config \
    patchelf \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Copy only requirements first — cache-friendly
COPY requirements.txt .

# Upgrade pip/setuptools/wheel and install dependencies
RUN pip install --upgrade pip setuptools wheel build \
    && pip install --timeout=1200 --use-pep517 -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH to /app so all projects are importable
ENV PYTHONPATH=/app

EXPOSE 8000

# Default command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
