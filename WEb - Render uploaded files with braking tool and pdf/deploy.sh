#!/bin/bash
# Render Platform Deployment Script

echo "Starting deployment on Render platform..."

# Check if we're in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    export IS_DOCKER=true
else
    echo "Running in native environment"
fi

# Check LaTeX installation
echo "Checking LaTeX engines..."
which pdflatex && echo "pdflatex: OK" || echo "pdflatex: NOT FOUND"
which xelatex && echo "xelatex: OK" || echo "xelatex: NOT FOUND"
which lualatex && echo "lualatex: OK" || echo "lualatex: NOT FOUND"

# Run the automated builder
echo "Running automated PDF builder..."
python docker_render_builder.py

# Start the web application
echo "Starting FastAPI application..."
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT
