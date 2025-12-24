FROM python:3.11-slim

# Install OS packages including comprehensive TeX Live for PDF generation
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       wget \
       git \
       fontconfig \
       poppler-utils \
       texlive-latex-recommended \
       texlive-latex-extra \
       texlive-fonts-recommended \
       texlive-fonts-extra \
       texlive-xetex \
       texlive-luatex \
       texlive-lang-english \
       texlive-science \
       texlive-pictures \
       libpq-dev \
       ghostscript \
       imagemagick \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy application files
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEXMFCACHE=/tmp/texmf-cache
ENV PATH="/usr/local/texlive/2023/bin/x86_64-linux:${PATH}"

# Create output directories
RUN mkdir -p /app/workspace_output/artifacts /app/workspace_output/build-logs

# Make sure the builder script is executable
RUN chmod +x /app/docker_render_builder.py || true

# Expose port
EXPOSE $PORT

# Expose port and run with gunicorn + uvicorn worker
EXPOSE 8000
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
