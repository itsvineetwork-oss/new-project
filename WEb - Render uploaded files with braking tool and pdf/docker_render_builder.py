#!/usr/bin/env python3
"""
Docker-Ready Automated Builder for Render Platform
Optimized for texlive PDF generation in container environment
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Check if we're running in Docker
IS_DOCKER = os.path.exists('/.dockerenv') or 'KUBERNETES_' in os.environ

class RenderDockerBuilder:
    def __init__(self):
        # Set paths based on environment
        if IS_DOCKER:
            # In Docker container on Render
            self.workspace_path = Path('/app')
            self.output_path = Path('/app/workspace_output')
        else:
            # Local development
            self.workspace_path = Path('.')
            self.output_path = Path('./workspace_output')
        
        self.build_logs = []
        self.actions_taken = []
        self.artifacts = []
        self.status = "success"
        
        # Ensure output directories exist
        for subdir in ['build-logs', 'changes', 'artifacts']:
            (self.output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def log_action(self, action_type, description, status="success", details=None):
        """Log an action taken during the build process."""
        action = {
            "timestamp": datetime.now().isoformat(),
            "type": action_type,
            "description": description,
            "status": status,
            "details": details or {},
            "environment": "docker" if IS_DOCKER else "local"
        }
        self.actions_taken.append(action)
        print(f"[{action_type.upper()}] {description} - {status}")
    
    def run_command(self, cmd, cwd=None, log_file=None):
        """Run a command and capture output."""
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.workspace_path,
                capture_output=True,
                text=True,
                shell=True,
                timeout=300  # 5 minute timeout
            )
            
            if log_file:
                log_path = self.output_path / "build-logs" / log_file
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Command: {cmd}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Environment: {'Docker' if IS_DOCKER else 'Local'}\n")
                    f.write(f"CWD: {cwd or self.workspace_path}\n")
                    f.write(f"STDOUT:\n{result.stdout}\n")
                    f.write(f"STDERR:\n{result.stderr}\n")
            
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            error_msg = f"Command timeout after 300s: {cmd}"
            if log_file:
                log_path = self.output_path / "build-logs" / log_file
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Command: {cmd}\n")
                    f.write(f"Error: {error_msg}\n")
            return False, "", error_msg
        except Exception as e:
            if log_file:
                log_path = self.output_path / "build-logs" / log_file
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Command: {cmd}\n")
                    f.write(f"Error: {str(e)}\n")
            return False, "", str(e)
    
    def check_tex_engines(self):
        """Check which LaTeX engines are available."""
        engines = ["pdflatex", "xelatex", "lualatex"]
        available_engines = []
        
        for engine in engines:
            success, _, _ = self.run_command(f"which {engine}")
            if success:
                available_engines.append(engine)
                self.log_action("tex_check", f"Found {engine}")
        
        return available_engines
    
    def build_latex_pdf(self, tex_file, max_attempts=3):
        """Build LaTeX file to PDF using available engines."""
        available_engines = self.check_tex_engines()
        
        if not available_engines:
            self.log_action("latex_error", f"No LaTeX engines available for {tex_file.name}", "error")
            return False
        
        # Prepare tex file for compilation
        tex_dir = tex_file.parent
        tex_name = tex_file.stem
        
        for engine in available_engines:
            self.log_action("latex_compile", f"Trying {engine} for {tex_file.name}")
            
            # Copy logo files if they don't exist in tex directory
            logo_files = ["logo.JPG", "logo-1.JPG", "logo.png"]
            for logo in logo_files:
                src_logo = self.workspace_path / "working copy Docker web with braking" / logo
                dst_logo = tex_dir / logo
                if src_logo.exists() and not dst_logo.exists():
                    try:
                        shutil.copy2(src_logo, dst_logo)
                        self.log_action("file_copy", f"Copied {logo} to tex directory")
                    except Exception as e:
                        self.log_action("file_copy_error", f"Failed to copy {logo}: {str(e)}", "error")
            
            # Run LaTeX compilation multiple times for references
            for attempt in range(max_attempts):
                cmd = f"{engine} -interaction=nonstopmode -output-directory={tex_dir} {tex_name}.tex"
                success, stdout, stderr = self.run_command(
                    cmd, 
                    cwd=tex_dir,
                    log_file=f"{tex_name}-{engine}-attempt{attempt+1}.log"
                )
                
                if not success:
                    self.log_action("latex_attempt", f"{engine} attempt {attempt+1} failed", "error")
                    if attempt == max_attempts - 1:  # Last attempt with this engine
                        break
                else:
                    self.log_action("latex_attempt", f"{engine} attempt {attempt+1} succeeded")
            
            # Check if PDF was created
            pdf_file = tex_dir / f"{tex_name}.pdf"
            if pdf_file.exists():
                # Move PDF to artifacts
                artifact_path = self.output_path / "artifacts" / f"{tex_name}_{engine}.pdf"
                shutil.copy2(pdf_file, artifact_path)
                
                self.artifacts.append({
                    "source": str(tex_file),
                    "output": str(artifact_path),
                    "type": "pdf",
                    "engine": engine,
                    "size": artifact_path.stat().st_size
                })
                self.log_action("latex_success", f"Generated PDF with {engine}: {artifact_path.name}")
                return True
            else:
                self.log_action("latex_no_pdf", f"{engine} completed but no PDF generated", "error")
        
        return False
    
    def process_latex_files(self):
        """Process all LaTeX files in the workspace."""
        tex_files = list(self.workspace_path.rglob("*.tex"))
        
        if not tex_files:
            self.log_action("latex_scan", "No LaTeX files found")
            return
        
        self.log_action("latex_scan", f"Found {len(tex_files)} LaTeX files")
        
        for tex_file in tex_files:
            self.log_action("latex_process", f"Processing {tex_file.name}")
            success = self.build_latex_pdf(tex_file)
            
            if not success:
                # Fallback: create a simple text version
                try:
                    with open(tex_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract title and content
                    lines = content.split('\n')
                    text_content = []
                    for line in lines:
                        if '\\title{' in line:
                            title = line.replace('\\title{', '').replace('}', '')
                            text_content.append(f"TITLE: {title}")
                        elif '\\section{' in line:
                            section = line.replace('\\section{', '').replace('}', '')
                            text_content.append(f"\nSECTION: {section}")
                        elif not line.strip().startswith('\\') and line.strip():
                            text_content.append(line)
                    
                    # Save as text file
                    txt_path = self.output_path / "artifacts" / f"{tex_file.stem}_fallback.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(text_content))
                    
                    self.artifacts.append({
                        "source": str(tex_file),
                        "output": str(txt_path),
                        "type": "text_fallback",
                        "size": txt_path.stat().st_size
                    })
                    self.log_action("latex_fallback", f"Created text fallback: {txt_path.name}")
                    
                except Exception as e:
                    self.log_action("latex_fallback_error", f"Failed to create fallback: {str(e)}", "error")
    
    def create_deployment_script(self):
        """Create a deployment script for Render platform."""
        deploy_script = """#!/bin/bash
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
"""
        
        script_path = self.workspace_path / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(deploy_script)
        
        # Make executable
        script_path.chmod(0o755)
        
        self.log_action("deploy_script", "Created deployment script")
    
    def generate_report(self):
        """Generate comprehensive build report."""
        # Count files by type
        file_counts = {
            "tex": len(list(self.workspace_path.rglob("*.tex"))),
            "py": len(list(self.workspace_path.rglob("*.py"))),
            "html": len(list(self.workspace_path.rglob("*.html"))),
            "md": len(list(self.workspace_path.rglob("*.md")))
        }
        
        # Determine status
        error_actions = [a for a in self.actions_taken if a["status"] == "error"]
        if error_actions:
            if len(error_actions) < len(self.actions_taken) / 2:
                self.status = "partial_success"
            else:
                self.status = "failed"
        
        report = {
            "build_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": "docker" if IS_DOCKER else "local",
                "platform": "render" if IS_DOCKER else "development"
            },
            "file_counts": file_counts,
            "actions_taken": self.actions_taken,
            "artifacts": self.artifacts,
            "logs": [str(f) for f in (self.output_path / "build-logs").glob("*")],
            "status": self.status,
            "summary": {
                "total_actions": len(self.actions_taken),
                "successful_actions": len([a for a in self.actions_taken if a["status"] == "success"]),
                "failed_actions": len(error_actions),
                "artifacts_generated": len(self.artifacts)
            }
        }
        
        report_path = self.output_path / "render_build_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Also create a summary for logging
        summary = f"""
🚀 Render Docker Build Summary
==============================
Environment: {'Docker/Render' if IS_DOCKER else 'Local Development'}
Status: {self.status.upper()}
Actions: {len([a for a in self.actions_taken if a['status'] == 'success'])}/{len(self.actions_taken)} successful
Artifacts: {len(self.artifacts)} generated

Files Processed:
- LaTeX: {file_counts['tex']} files
- Python: {file_counts['py']} files
- HTML: {file_counts['html']} files
- Markdown: {file_counts['md']} files

Output Directory: {self.output_path}
Report: {report_path}
"""
        
        print(summary)
        return report
    
    def run_build(self):
        """Execute the complete build process."""
        self.log_action("start", f"Starting Render Docker build in {'container' if IS_DOCKER else 'local'} environment")
        
        # Process LaTeX files (main focus)
        self.process_latex_files()
        
        # Create deployment script
        self.create_deployment_script()
        
        # Generate final report
        report = self.generate_report()
        
        self.log_action("complete", f"Build completed with status: {self.status}")
        return report

def main():
    builder = RenderDockerBuilder()
    report = builder.run_build()
    return report

if __name__ == "__main__":
    main()