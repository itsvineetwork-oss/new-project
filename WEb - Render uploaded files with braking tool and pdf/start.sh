#!/bin/bash
echo "🚀 Starting Premnath Rail Engineering Calculator on Render Platform"
echo "================================================="

# Check environment
echo "Environment Check:"
echo "- Docker: $([ -f /.dockerenv ] && echo 'YES' || echo 'NO')"
echo "- Port: ${PORT:-8000}"

# Check LaTeX installation
echo ""
echo "LaTeX Engines Check:"
which pdflatex > /dev/null && echo "✅ pdflatex: AVAILABLE" || echo "❌ pdflatex: NOT FOUND"
which xelatex > /dev/null && echo "✅ xelatex: AVAILABLE" || echo "❌ xelatex: NOT FOUND" 
which lualatex > /dev/null && echo "✅ lualatex: AVAILABLE" || echo "❌ lualatex: NOT FOUND"

# Run automated build once on startup to generate PDFs
echo ""
echo "🔧 Running automated PDF builder..."
python docker_render_builder.py || echo "⚠️ PDF builder completed with warnings"

# Create output directories if they don't exist
mkdir -p workspace_output/artifacts workspace_output/build-logs

# Start the FastAPI web application
echo ""
echo "🌐 Starting FastAPI application on port ${PORT:-8000}..."
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}