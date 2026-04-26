# Clinical Robot Adaptation Dockerfile
# Multi-stage build for production deployment

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set labels
LABEL maintainer="idriss.djiofack@colorado.edu" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="clinical-robot-adaptation" \
      org.label-schema.description="Clinical Robot Adaptation through Few-Shot Foundation Model Fine-Tuning" \
      org.label-schema.url="https://github.com/idrissdjio/Clinical-Robot-Adaptation" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/idrissdjio/Clinical-Robot-Adaptation.git" \
      org.label-schema.vendor="HIRO Laboratory, University of Colorado Boulder" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash clinical && \
    chown -R clinical:clinical /app

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH="/app" \
    PATH="/home/clinical/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    libhdf5-100 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-155 \
    libgtk-3-0 \
    libatlas3-base \
    libopenblas-pthread-dev \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash clinical && \
    chown -R clinical:clinical /app

# Switch to non-root user
USER clinical

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/results /app/.clinical_robot

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "api.fastapi_server"]

# Development variant
FROM production as development

# Switch to root user for development setup
USER root

# Install development dependencies
RUN pip install --no-cache-dir jupyterlab ipython

# Switch back to clinical user
USER clinical

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Testing variant
FROM production as testing

# Install testing dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-mock pytest-benchmark

# Copy test files
COPY --from=builder /app/tests /app/tests

# Test command
CMD ["pytest", "tests/", "-v", "--cov=/app", "--cov-report=html", "--cov-report=xml"]
