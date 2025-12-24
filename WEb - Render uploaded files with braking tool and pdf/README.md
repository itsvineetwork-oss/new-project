# 🚀 Premnath Rail Engineering Calculator - Render Deployment

## 📁 Deployment Package Contents

This folder contains all necessary files for deploying the Premnath Rail Engineering Calculator on Render platform with Docker + LaTeX PDF generation.

### 🔧 Core Application Files
- `main.py` - FastAPI application (33 KB)
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration with LaTeX
- `start.sh` - Startup script for Render
- `docker_render_builder.py` - Automated PDF builder

### 🗄️ Database & Models
- `database.py` - Database connection
- `models.py` - SQLAlchemy models
- `schemas.py` - Pydantic schemas

### 📊 Calculator Logic (7 modules)
- `Hydraulic_Motor_Calculation.py` (34 KB)
- `QmaxCalculator_Logic.py`
- `LoadDistribution_Logic.py` 
- `Tractive_Effort_Logic.py`
- `Vehicle_Performance_Logic.py`
- `braking_logic.py`
- `braking.py` (115 KB)

### 🌐 Frontend HTML Pages (10 files)
- `index.html` - Homepage
- `login.html`, `signup.html` - Authentication
- `calculator.html` - Hydraulic calculator
- `qmax_calculator.html` - Qmax calculator
- `load_distribution_calculator.html`
- `tractive_effort_calculator.html`
- `vehicle_performance_calculator.html`
- `braking_calculator.html`
- `profile.html` - User profile

### 🖼️ Assets
- `logo.png` - Company logo (202 KB)
- `Diagram.png` - Technical diagram
- `template.tex` - LaTeX template for PDF reports (21 KB)

## 🚀 Render Deployment Steps

### 1. Create New Web Service
```
- Go to Render Dashboard
- Click "New" → "Web Service"
- Connect your GitHub repository
- Select branch: main
- Set Root Directory: deploy/ (if needed)
```

### 2. Build & Start Commands
```
Build Command: chmod +x start.sh
Start Command: ./start.sh
```

### 3. Environment Variables
```
PORT: (Auto-set by Render)
TEXMFCACHE: /tmp/texmf-cache
```

### 4. Features Included
✅ **LaTeX PDF Generation** - Professional engineering reports  
✅ **Multiple LaTeX Engines** - pdflatex, xelatex, lualatex  
✅ **Auto-fallback** - ReportLab if LaTeX unavailable  
✅ **Docker Optimization** - Fast startup with pre-built images  
✅ **Engineering Calculators** - 7 different calculation modules  

## 📊 Total Package Size
- **28 files** total
- **~580 KB** combined size
- Ready for production deployment

## 🔧 Local Testing (Optional)
```bash
# Test Docker build locally
docker build -t premnath-calc .
docker run -p 8000:8000 premnath-calc

# Access at http://localhost:8000
```

## 📞 Support
For deployment issues, contact: Premnath Engineering Works

---
**Generated on:** December 1, 2025  
**Package Version:** 1.0.0