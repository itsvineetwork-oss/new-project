import math
import io
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# --- Database & Security Imports ---
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import models
import schemas
from database import engine, get_db

# Optional imports
try:
    import docx
except ImportError:
    docx = None

import subprocess
import tempfile
import zipfile
import shutil
from datetime import datetime

# --- Import Logic ---
from Hydraulic_Motor_Calculation import Calculator as HydraulicCalculator
from QmaxCalculator_Logic import QmaxCalculatorLogic, SIGMA_B_OPTIONS
from LoadDistribution_Logic import perform_calculations as perform_load_distro_calc
from LoadDistribution_Logic import format_detailed_steps as format_load_distro_steps
from LoadDistribution_Logic import create_report_docx as create_load_distro_docx
from Tractive_Effort_Logic import perform_te_calculations, format_te_report_text, create_te_report_docx
from Vehicle_Performance_Logic import VehiclePerformanceCalculator
from braking_logic import perform_calculation_sequence

# --- DATABASE SETUP ---
models.Base.metadata.create_all(bind=engine)

# --- SECURITY SETUP ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# ---------- Pydantic Models ----------
class HydraulicRawInput(BaseModel):
    calc_mode: str
    weight: str
    axles: str
    speed: str
    max_vehicle_rpm: str
    pto_gear_ratio: str
    engine_gear_ratio: str
    axle_gear_box_ratio: str
    slope_percent: str
    curve_degree: str
    wheel_diameter: str
    num_motors: str
    per_axle_motor: str
    pressure: str
    mech_eff_motor: str
    vol_eff_motor: str
    motor_disp_in: str
    max_motor_rpm: str
    vol_eff_pump: str
    pump_disp_in: str
    max_pump_rpm: str

class QmaxRawInput(BaseModel):
    d: str
    sigma_b_selection: str
    sigma_b_custom: str
    v_head: str

class LoadDistroRawInput(BaseModel):
    config_type: str
    total_load: str
    front_percent: str
    q1_percent: str
    q3_percent: str

class TractiveEffortRawInput(BaseModel):
    load: str
    loco_weight: str
    gradient: str
    curvature: str
    speed: str
    mode: str
    grad_type: str
    curvature_unit: str

class VehiclePerformanceRawInput(BaseModel):
    max_curve: str
    max_slope: str
    loco_gvw: str
    max_speed: str
    num_axles: str
    rear_axle_ratio: str
    gear_ratios: str
    shunting_load: str
    peak_power: str
    friction_mu: str
    wheel_dia: str
    min_rpm: str
    max_rpm: str
    torque_curve: Dict[str, float] = Field(default_factory=dict)

class BrakingRawInput(BaseModel):
    mass_kg: float
    reaction_time: float
    num_wheels: int  # HTML sends 'num_wheels'
    calc_mode: str  # "Rail" or "Rail+Road"
    
    rail_speed_input: str  # Comma-separated speeds
    rail_gradient_input: str  # Comma-separated gradients
    rail_gradient_type: str  # "Degree (°)", "1 in G", or "Percentage (%)"
    
    road_speed_input: Optional[str] = ""
    road_gradient_input: Optional[str] = ""
    road_gradient_type: Optional[str] = "Percentage (%)"
    mu: Optional[float] = 0.7
    
    doc_no: Optional[str] = ""
    made_by: Optional[str] = ""
    checked_by: Optional[str] = ""
    approved_by: Optional[str] = ""
    wheel_dia: Optional[float] = 0

# ---------- Validation Functions ----------
def _validate_input(value_str: str, type_func, name: str, is_optional=False, default=0.0, is_disabled=False):
    if is_disabled: return default
    if not value_str:
        if is_optional: return default
        else: raise ValueError(f"'{name}' cannot be empty.")
    try:
        return type_func(value_str)
    except:
        raise ValueError(f"Invalid value for '{name}'")

# (Shortened validation wrappers for brevity - logic remains same as before)
def process_and_validate_hydraulic_inputs(raw: HydraulicRawInput):
    # ... (Your existing validation logic) ...
    # NOTE: Main logic wahi hai jo pichli baar tha, bas space bachane ke liye yahan short mein likh raha hu
    # Aap purana validation logic as-is rakh sakte hain
    # Lekin agar aap chahein to main poora function dobara likh deta hu:
    inputs = {}
    inputs_raw = raw.dict()
    mode = raw.calc_mode
    inputs['calc_mode'] = mode
    inputs['weight'] = _validate_input(raw.weight, float, "Vehicle Weight")
    inputs['axles'] = _validate_input(raw.axles, int, "Number of axles")
    inputs['num_motors'] = _validate_input(raw.num_motors, int, "Hydraulic Motor")
    inputs['per_axle_motor'] = _validate_input(raw.per_axle_motor, int, "Motor / axle")
    inputs['slope_percent'] = _validate_input(raw.slope_percent, float, "Slope", True)
    inputs['curve_degree'] = _validate_input(raw.curve_degree, float, "Curve", True)
    inputs['wheel_diameter'] = _validate_input(raw.wheel_diameter, float, "Wheel Dia")
    inputs['axle_gear_box_ratio'] = _validate_input(raw.axle_gear_box_ratio, float, "Axle Gear Ratio")
    
    engine_gear_raw = raw.engine_gear_ratio.strip()
    parts = [p.strip() for p in engine_gear_raw.split(',') if p.strip()]
    engine_gear_list = [_validate_input(p, float, "Engine Gear") for p in parts]
    inputs['engine_gear_ratio_list'] = engine_gear_list
    inputs['engine_gear_ratio'] = engine_gear_list[0] if engine_gear_list else 1.0
    
    inputs['max_vehicle_rpm'] = _validate_input(raw.max_vehicle_rpm, float, "Max RPM")
    inputs['pto_gear_ratio'] = _validate_input(raw.pto_gear_ratio, float, "PTO Ratio")
    inputs['vol_eff_motor'] = _validate_input(raw.vol_eff_motor, float, "Motor Vol Eff")
    inputs['vol_eff_pump'] = _validate_input(raw.vol_eff_pump, float, "Pump Vol Eff")
    
    is_cc = (mode == 'calc_cc')
    inputs['speed'] = _validate_input(raw.speed, float, "Speed", True, 0.0, not is_cc)
    inputs['pressure'] = _validate_input(raw.pressure, float, "Pressure", True, 0.0, not is_cc)
    inputs['mech_eff_motor'] = _validate_input(raw.mech_eff_motor, float, "Mech Eff", True, 0.0, not is_cc)
    inputs['motor_disp_in'] = _validate_input(raw.motor_disp_in, float, "Motor Disp", True, 0.0, is_cc)
    inputs['max_motor_rpm'] = _validate_input(raw.max_motor_rpm, float, "Max Motor RPM", True, 3000.0, is_cc)
    inputs['pump_disp_in'] = _validate_input(raw.pump_disp_in, float, "Pump Disp", True, 0.0, is_cc)
    inputs['max_pump_rpm'] = _validate_input(raw.max_pump_rpm, float, "Max Pump RPM", True, 3000.0, is_cc)
    
    return inputs, inputs_raw

def process_and_validate_qmax_inputs(raw: QmaxRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['d'] = _validate_input(raw.d, float, "Worn rail diameter")
    inputs['v_head'] = _validate_input(raw.v_head, float, "Safety Factor")
    if raw.sigma_b_selection == "Custom":
        inputs['sigma_b'] = _validate_input(raw.sigma_b_custom, float, "Custom Sigma B")
    else:
        inputs['sigma_b'] = SIGMA_B_OPTIONS.get(raw.sigma_b_selection, 0)
    return inputs, inputs_raw

def process_and_validate_load_distro_inputs(raw: LoadDistroRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['config_type'] = raw.config_type
    inputs['total_load'] = _validate_input(raw.total_load, float, "Total Load")
    inputs['front_percent'] = _validate_input(raw.front_percent, float, "Front %")
    inputs['q1_percent'] = _validate_input(raw.q1_percent, float, "Q1 %")
    inputs['q3_percent'] = _validate_input(raw.q3_percent, float, "Q3 %")
    return inputs, inputs_raw

def process_and_validate_te_inputs(raw: TractiveEffortRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['load'] = _validate_input(raw.load, float, "Load")
    inputs['loco_weight'] = _validate_input(raw.loco_weight, float, "Loco Weight")
    inputs['gradient'] = _validate_input(raw.gradient, float, "Gradient")
    inputs['curvature'] = _validate_input(raw.curvature, float, "Curvature")
    inputs['speed'] = _validate_input(raw.speed, float, "Speed")
    inputs['mode'] = raw.mode
    inputs['grad_type'] = raw.grad_type
    inputs['curvature_unit'] = raw.curvature_unit
    return inputs, inputs_raw

def process_and_validate_vehicle_performance_inputs(raw: VehiclePerformanceRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['max_curve'] = _validate_input(raw.max_curve, float, "Max Curve", True)
    inputs['curve_unit'] = 'degree'
    inputs['max_slope'] = _validate_input(raw.max_slope, float, "Max Slope", True)
    inputs['slope_unit'] = '%'
    inputs['loco_gvw_kg'] = _validate_input(raw.loco_gvw, float, "Loco GVW")
    inputs['max_speed_kmh'] = _validate_input(raw.max_speed, float, "Max Speed", True)
    inputs['num_axles'] = _validate_input(raw.num_axles, int, "Axles")
    inputs['rear_axle_ratio'] = _validate_input(raw.rear_axle_ratio, float, "Rear Ratio")
    
    ratios = (raw.gear_ratios or "").split(',')
    inputs['gear_ratios'] = [_validate_input(r.strip(), float, "Gear Ratio") for r in ratios if r.strip()]
    if not inputs['gear_ratios']: inputs['gear_ratios'] = [1.0]
    
    inputs['shunting_load_t'] = _validate_input(raw.shunting_load, float, "Shunting Load", True)
    inputs['peak_power_kw'] = _validate_input(raw.peak_power, float, "Peak Power", True)
    inputs['friction_mu'] = _validate_input(raw.friction_mu, float, "Friction")
    inputs['wheel_dia_m'] = _validate_input(raw.wheel_dia, float, "Wheel Dia")
    inputs['min_rpm'] = _validate_input(raw.min_rpm, int, "Min RPM", True)
    inputs['max_rpm'] = _validate_input(raw.max_rpm, int, "Max RPM")
    
    tc = {}
    if raw.torque_curve:
        for k, v in raw.torque_curve.items():
            try: tc[int(k)] = float(v)
            except: pass
    inputs['torque_curve'] = tc
    return inputs, inputs_raw

def process_and_validate_braking_inputs(raw: BrakingRawInput):
    inputs = {}
    inputs_raw = raw.dict()
    inputs['mass_kg'] = raw.mass_kg
    inputs['speed_kmh'] = raw.speed_kmh
    inputs['mu'] = raw.mu
    inputs['reaction_time'] = raw.reaction_time
    inputs['gradient'] = raw.gradient
    inputs['gradient_type'] = raw.gradient_type
    inputs['num_wheels'] = raw.num_wheels
    inputs['speed_increment'] = raw.speed_increment if raw.speed_increment else 10.0
    inputs['gradient_steps'] = raw.gradient_steps if raw.gradient_steps else 5
    
    # Basic validation
    if inputs['mass_kg'] <= 0:
        raise ValueError("Mass must be greater than 0")
    if inputs['num_wheels'] <= 0:
        raise ValueError("Number of wheels must be greater than 0")
    if inputs['mu'] <= 0:
        raise ValueError("Friction coefficient must be greater than 0")
    
    return inputs, inputs_raw


# =======================================================
# ⚠️ CRITICAL: APP INITIALIZATION MUST BE HERE (BEFORE ROUTES)
# =======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Logic Instances
hydraulic_calculator = HydraulicCalculator()
qmax_calculator = QmaxCalculatorLogic()


# ==========================================
#      STATIC PAGE ROUTES (HTML Serving)
# ==========================================

@app.get("/")
async def serve_home():
    return FileResponse('index.html')

@app.get("/braking_calculator.html")
async def serve_braking_calculator():
    return FileResponse('braking_calculator.html')

@app.get("/login")
async def serve_login_page():
    if os.path.exists("login.html"):
        return FileResponse('login.html')
    return {"error": "login.html file not found on server"}

@app.get("/signup")
async def serve_signup_page():
    if os.path.exists("signup.html"):
        return FileResponse('signup.html')
    return {"error": "signup.html file not found on server"}

@app.get("/profile")
async def serve_profile_page():
    if os.path.exists("profile.html"):
        return FileResponse('profile.html')
    return {"error": "profile.html file not found on server"}

@app.get("/calculator")
async def serve_hydraulic_calculator():
    return FileResponse('calculator.html')

@app.get("/qmax")
async def serve_qmax_page():
    return FileResponse('qmax_calculator.html')

@app.get("/load_distribution")
async def serve_load_distribution_page():
    return FileResponse('load_distribution_calculator.html')

@app.get("/tractive_effort")
async def serve_tractive_effort_page():
    return FileResponse('tractive_effort_calculator.html')

@app.get("/vehicle_performance")
async def serve_vehicle_performance_page():
    return FileResponse('vehicle_performance_calculator.html')

@app.get("/braking")
async def serve_braking_calculator_page():
    return FileResponse('braking_calculator.html')

@app.get("/Diagram.png")
async def serve_diagram():
    if os.path.exists("Diagram.png"): return FileResponse('Diagram.png')
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/logo.png")
async def serve_logo():
    if os.path.exists("logo.png"): return FileResponse('logo.png')
    raise HTTPException(status_code=404, detail="Not found")


# ==========================================
#      AUTH & API ROUTES
# ==========================================

@app.post("/signup")
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Account created successfully", "user_id": new_user.id}

@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {
        "message": "Login successful",
        "user_id": db_user.id,
        "email": db_user.email,
        "is_license_active": db_user.is_license_active
    }

@app.post("/activate_license")
def activate_license(activation: schemas.LicenseActivate, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    
    license_entry = db.query(models.LicenseKey).filter(models.LicenseKey.key == activation.license_key).first()
    if not license_entry: raise HTTPException(status_code=400, detail="Invalid License Key")
    if license_entry.is_used: raise HTTPException(status_code=400, detail="Key already used")
        
    license_entry.is_used = True
    user.is_license_active = True
    db.commit()
    return {"message": "License Activated!"}

@app.post("/generate_key")
def generate_license_key(key_code: str, db: Session = Depends(get_db)):
    existing = db.query(models.LicenseKey).filter(models.LicenseKey.key == key_code).first()
    if existing: raise HTTPException(status_code=400, detail="Key exists")
    new_key = models.LicenseKey(key=key_code)
    db.add(new_key)
    db.commit()
    return {"message": f"Key '{key_code}' created"}


# ==========================================
#      CALCULATION ROUTES
# ==========================================

@app.post("/calculate")
async def handle_calculation(raw_input: HydraulicRawInput):
    try:
        inputs, inputs_raw = process_and_validate_hydraulic_inputs(raw_input)
        hydraulic_calculator.inputs_raw = inputs_raw
        if inputs['calc_mode'] == "calc_cc":
            res = hydraulic_calculator.perform_displacement_calculation(inputs)
            rep = hydraulic_calculator._generate_mode1_report(inputs, res)
        else:
            res = hydraulic_calculator.perform_speed_calculation(inputs)
            rep = hydraulic_calculator._generate_mode2_report(inputs, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_report")
async def download_report(raw_input: HydraulicRawInput):
    if docx is None: raise HTTPException(500, "python-docx not installed")
    try:
        inputs, inputs_raw = process_and_validate_hydraulic_inputs(raw_input)
        hydraulic_calculator.inputs_raw = inputs_raw
        doc = docx.Document()
        if inputs['calc_mode'] == "calc_cc":
            res = hydraulic_calculator.perform_displacement_calculation(inputs)
            hydraulic_calculator._create_mode1_docx(doc, inputs, res)
            fname = "Hydraulic_Report.docx"
        else:
            res = hydraulic_calculator.perform_speed_calculation(inputs)
            hydraulic_calculator._create_mode2_docx(doc, inputs, res)
            fname = "Speed_Report.docx"
        stream = io.BytesIO()
        doc.save(stream)
        stream.seek(0)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_qmax")
async def handle_qmax(raw: QmaxRawInput):
    try:
        inp, raw_inp = process_and_validate_qmax_inputs(raw)
        qmax_calculator.inputs_raw = raw_inp
        res = qmax_calculator.perform_calculations(inp['d'], inp['sigma_b'], inp['v_head'])
        rep = qmax_calculator.format_detailed_steps(res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_qmax_report")
async def dl_qmax(raw: QmaxRawInput):
    try:
        inp, raw_inp = process_and_validate_qmax_inputs(raw)
        qmax_calculator.inputs_raw = raw_inp
        res = qmax_calculator.perform_calculations(inp['d'], inp['sigma_b'], inp['v_head'])
        stream = qmax_calculator.create_report_docx(res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Qmax_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_load_distribution")
async def handle_load(raw: LoadDistroRawInput):
    try:
        inp, raw_inp = process_and_validate_load_distro_inputs(raw)
        res = perform_load_distro_calc(inp['config_type'], inp['total_load'], inp['front_percent'], inp['q1_percent'], inp['q3_percent'])
        rep = format_load_distro_steps(inp, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_load_distribution_report")
async def dl_load(raw: LoadDistroRawInput):
    try:
        inp, raw_inp = process_and_validate_load_distro_inputs(raw)
        res = perform_load_distro_calc(inp['config_type'], inp['total_load'], inp['front_percent'], inp['q1_percent'], inp['q3_percent'])
        stream = create_load_distro_docx(inp, res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Load_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_tractive_effort")
async def handle_te(raw: TractiveEffortRawInput):
    try:
        inp, raw_inp = process_and_validate_te_inputs(raw)
        res = perform_te_calculations(inp)
        rep = format_te_report_text(inp, res)
        return {"report": rep, "results": res}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_tractive_effort_report")
async def dl_te(raw: TractiveEffortRawInput):
    try:
        inp, raw_inp = process_and_validate_te_inputs(raw)
        res = perform_te_calculations(inp)
        stream = create_te_report_docx(inp, res)
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=TE_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/calculate_performance")
async def handle_perf(raw: VehiclePerformanceRawInput):
    try:
        inp, raw_inp = process_and_validate_vehicle_performance_inputs(raw)
        calc = VehiclePerformanceCalculator(inp)
        res = calc.run_tractive_calculation()
        plot = calc.calculate_plot_data()
        table = calc.calculate_speed_for_shunting_load()
        return {"traction_snapshot": res, "tractive_effort_graph": plot["tractive_effort_plot"], "shunting_capability_graph": plot["shunting_capability_plot"], "speed_vs_slope_table": table}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/download_performance_report")
async def dl_perf(raw: VehiclePerformanceRawInput):
    try:
        inp, raw_inp = process_and_validate_vehicle_performance_inputs(raw)
        calc = VehiclePerformanceCalculator(inp)
        stream = calc.create_report_docx()
        return StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Performance_Report.docx"})
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/braking_calculate")
async def handle_braking_calculation(raw: BrakingRawInput):
    try:
        inputs = raw.dict()
        results_table_rows, context = perform_calculation_sequence(inputs)
        return {"rows": results_table_rows}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/download_braking_report")
async def download_braking_report(raw: BrakingRawInput):
    print("=" * 80)
    print("DEBUG: download_braking_report ENDPOINT CALLED!")
    print("=" * 80)
    try:
        inputs = raw.dict()
        print(f"DEBUG: Received inputs: {inputs}")
        results_table_rows, context = perform_calculation_sequence(inputs)
        print(f"DEBUG: Context keys: {list(context.keys())}")
        
        # DEBUG: Print some key context values
        print(f"DEBUG: mass_kg = {context.get('mass_kg', 'MISSING')}")
        print(f"DEBUG: speed_kmh = {context.get('speed_kmh', 'MISSING')}")
        print(f"DEBUG: fb = {context.get('fb', 'MISSING')}")
        print(f"DEBUG: max_braking_force = {context.get('max_braking_force', 'MISSING')}")
        print(f"DEBUG: rail_detailed_calcs count = {len(context.get('rail_detailed_calcs', []))}")
        print(f"DEBUG: old_data_for_report count = {len(context.get('old_data_for_report', {}))}")
        
        # Check if we're in Docker and have LaTeX available
        is_docker = os.path.exists('/.dockerenv') or 'KUBERNETES_' in os.environ
        
        # Try LaTeX with Jinja2 template (prefer this in Docker)
        if is_docker:
            try:
                # Run the docker render builder to generate PDFs
                subprocess.run([
                    'python', 'docker_render_builder.py'
                ], check=True, timeout=60)
                print("DEBUG: Docker builder completed successfully")
                
                # Check if PDF was generated
                pdf_path = 'workspace_output/artifacts/template_pdflatex.pdf'
                if not os.path.exists(pdf_path):
                    pdf_path = 'workspace_output/artifacts/template_xelatex.pdf'
                if not os.path.exists(pdf_path):
                    pdf_path = 'workspace_output/artifacts/template_lualatex.pdf'
                if not os.path.exists(pdf_path):
                    pdf_path = 'workspace_output/artifacts/template_converted.pdf'
                
                if os.path.exists(pdf_path):
                    with open(pdf_path, 'rb') as f:
                        pdf_content = f.read()
                    return StreamingResponse(
                        io.BytesIO(pdf_content),
                        media_type="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=Braking_Performance_Report.pdf"}
                    )
            except Exception as docker_error:
                print(f"DEBUG: Docker builder failed: {str(docker_error)}")
        
        # Fallback: Try local LaTeX compilation
        try:
            from jinja2 import Environment, FileSystemLoader
            import shutil
            
            # Set up Jinja2 environment to load template.tex
            env = Environment(loader=FileSystemLoader('.'))
            template = env.get_template('template.tex')
            
            # Render the template with context data
            print("DEBUG: Starting template rendering...")
            print(f"DEBUG: Context keys: {list(context.keys())}")
            try:
                rendered_tex = template.render(context)
                print(f"DEBUG: Template rendered successfully. Length: {len(rendered_tex)}")
                print(f"DEBUG: Template preview: {rendered_tex[:200]}...")
            except Exception as render_err:
                print(f"DEBUG: Template rendering FAILED!")
                print(f"DEBUG: Error type: {type(render_err).__name__}")
                print(f"DEBUG: Error message: {str(render_err)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, "braking_report.tex")
                pdf_file = os.path.join(tmpdir, "braking_report.pdf")
                
                # Copy image files if they exist (more robust)
                print(f"DEBUG: Looking for image files...")
                logo_mappings = {
                    'logo.JPG': [ 'logo.jpg', 'logo.JPG', 'Diagram.png'],
                    'logo-1.JPG': ['logo-1.png', 'logo-1.jpg', 'logo-1.JPG', 'logo.png'],
                    'breaking distance table.png': ['breaking distance table.png', 'braking_distance_table.png']
                }
                
                for target, sources in logo_mappings.items():
                    copied = False
                    for src in sources:
                        if os.path.exists(src):
                            try:
                                shutil.copy(src, os.path.join(tmpdir, target))
                                print(f"DEBUG: Copied {src} -> {target}")
                                copied = True
                                break
                            except Exception as e:
                                print(f"DEBUG: Failed to copy {src}: {e}")
                    if not copied:
                        print(f"DEBUG: No source found for {target}")
                
                # Write rendered LaTeX file
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(rendered_tex)
                print(f"DEBUG: Wrote .tex file to {tex_file}")
                print(f"DEBUG: LaTeX content length: {len(rendered_tex)} characters")
                
                # Try different LaTeX engines with better error handling
                engines = ['pdflatex', 'xelatex', 'lualatex']
                pdf_created = False
                
                for engine in engines:
                    try:
                        print(f"DEBUG: Trying {engine}...")
                        # Check if engine is available
                        try:
                            check_result = subprocess.run(
                                [engine, '--version'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                timeout=10
                            )
                            if check_result.returncode != 0:
                                print(f"DEBUG: {engine} not available")
                                continue
                        except FileNotFoundError:
                            print(f"DEBUG: {engine} not found")
                            continue
                            
                        # Use shorter file paths to avoid Windows path issues
                        simple_tex_name = "report.tex"
                        simple_tex_path = os.path.join(tmpdir, simple_tex_name)
                        simple_pdf_path = os.path.join(tmpdir, "report.pdf")
                        
                        # Copy tex file to simpler name
                        shutil.copy(tex_file, simple_tex_path)
                        
                        # Compile with LaTeX (run twice for proper formatting)
                        for run_num in range(2):
                            print(f"DEBUG: {engine} run {run_num + 1}/2...")
                            result_proc = subprocess.run(
                                [engine, '-interaction=nonstopmode', '-output-directory=.', simple_tex_name],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=60,
                                cwd=tmpdir
                            )
                            
                            if result_proc.returncode != 0:
                                print(f"DEBUG: {engine} run {run_num + 1} failed. Return code: {result_proc.returncode}")
                                print(f"DEBUG: STDOUT: {result_proc.stdout.decode('utf-8', errors='ignore')[:500]}...")
                                print(f"DEBUG: STDERR: {result_proc.stderr.decode('utf-8', errors='ignore')[:500]}...")
                                if run_num == 0:  # If first run fails, don't try second run
                                    break
                        
                        # Check if PDF was created
                        if os.path.exists(simple_pdf_path) and result_proc.returncode == 0:
                            # Copy back to expected location
                            shutil.copy(simple_pdf_path, pdf_file)
                            print(f"DEBUG: PDF created successfully with {engine}")
                            pdf_created = True
                            break
                        else:
                            print(f"DEBUG: PDF not created with {engine}. File exists: {os.path.exists(simple_pdf_path)}, Return code: {result_proc.returncode}")
                            
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                        print(f"DEBUG: {engine} failed with exception: {str(e)}")
                        continue
                
                if pdf_created:
                    with open(pdf_file, 'rb') as f:
                        pdf_content = f.read()
                    
                    print(f"DEBUG: Returning LaTeX-generated PDF ({len(pdf_content)} bytes)")
                    return StreamingResponse(
                        io.BytesIO(pdf_content),
                        media_type="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=Braking_Performance_Report_LaTeX.pdf"}
                    )
                else:
                    print(f"DEBUG: All LaTeX engines failed, falling back to ReportLab")
                    raise FileNotFoundError("PDF not generated by any LaTeX engine")
                    
        except Exception as latex_error:
            # Fallback to ReportLab if LaTeX fails
            print(f"DEBUG: LaTeX error: {type(latex_error).__name__}: {str(latex_error)}")
            
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e3a8a'),
                spaceAfter=12,
                alignment=TA_CENTER
            )
            story.append(Paragraph("Braking Performance Report", title_style))
            story.append(Paragraph("Based on DIN EN 15746-2:2021-05", styles['Heading2']))
            story.append(Spacer(1, 0.3*inch))
            
            # Get the first result for display (assuming single calculation)
            if results_table_rows:
                result = results_table_rows[0]
                
                # Input Parameters Section
                story.append(Paragraph("Input Parameters", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                input_data = [
                    ['Parameter', 'Value'],
                    ['Vehicle Mass', f"{inputs.get('mass_kg', 'N/A')} kg"],
                    ['Number of Wheels', f"{inputs.get('num_wheels', 'N/A')}"],
                    ['Speed', f"{inputs.get('speed_kmh', 'N/A')} km/h"],
                    ['Coefficient of Friction (μ)', f"{inputs.get('mu', 'N/A')}"],
                    ['Reaction Time', f"{inputs.get('reaction_time', 'N/A')} s"],
                    ['Gradient', f"{inputs.get('gradient', 'N/A')} ({inputs.get('gradient_type', 'N/A')})"],
                ]
                
                input_table = Table(input_data, colWidths=[3*inch, 3*inch])
                input_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(input_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Results Section
                story.append(Paragraph("Calculation Results", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                results_data = [
                    ['Parameter', 'Value'],
                    ['Braking Distance', f"{result.get('braking_distance_m', 'N/A')} m"],
                    ['Total Stopping Distance', f"{result.get('total_stopping_distance_m', 'N/A')} m"],
                    ['Deceleration', f"{result.get('deceleration_m_s2', 'N/A')} m/s²"],
                ]
                
                results_table = Table(results_data, colWidths=[3*inch, 3*inch])
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(results_table)
            
            # Build the document
            doc.build(story)
            buffer.seek(0)
            
            print(f"DEBUG: Returning ReportLab-generated PDF ({buffer.tell()} bytes)")
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=Braking_Performance_Report.pdf"}
            )
            
    except Exception as e:
        print(f"DEBUG: OUTER EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/download_braking_folder")
async def download_braking_folder(raw: BrakingRawInput):
    """
    Generate and download PDF directly
    """
    print("=" * 80)
    print("DEBUG: download_braking_folder ENDPOINT CALLED!")
    print("=" * 80)
    try:
        inputs = raw.dict()
        print(f"DEBUG: Received inputs for PDF download: {inputs}")
        results_table_rows, context = perform_calculation_sequence(inputs)
        
        # Create timestamp for unique folder names
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a temporary folder for PDF generation
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # Generate TEX content
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader('.'))
            template = env.get_template('template.tex')
            rendered_tex = template.render(context)
            
            # Save TEX file
            tex_file = os.path.join(tmpdir, 'Braking_Report.tex')
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(rendered_tex)
            print(f"DEBUG: TEX file saved: {tex_file}")
            
            # Copy required images for LaTeX compilation
            required_images = ['logo.png', 'logo.JPG', 'logo-1.JPG', 'breaking distance table.png', 
                             'gradient.jpg', 'curve.jpg', 'superelevation.png', 'cant.png', 
                             'gauge.jpg', 'disc_brake.jpg', 'tread_brake.png']
            
            for img_name in required_images:
                if os.path.exists(img_name):
                    try:
                        shutil.copy(img_name, os.path.join(tmpdir, img_name))
                        print(f"DEBUG: Copied image: {img_name}")
                    except Exception as e:
                        print(f"DEBUG: Failed to copy {img_name}: {e}")
            
            # Generate PDF
            pdf_generated = False
            pdf_file = os.path.join(tmpdir, 'Braking_Report.pdf')
            
            print(f"DEBUG: Attempting PDF generation...")
            try:
                # Check if pdflatex is available
                check_result = subprocess.run(
                    ['pdflatex', '--version'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=10
                )
                
                if check_result.returncode == 0:
                    print(f"DEBUG: pdflatex available, generating PDF...")
                    
                    # Run pdflatex twice for cross-references
                    for run_num in range(2):
                        print(f"DEBUG: Running pdflatex iteration {run_num + 1}...")
                        result_proc = subprocess.run(
                            ['pdflatex', '-interaction=nonstopmode', 'Braking_Report.tex'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=180,
                            cwd=tmpdir
                        )
                        
                        print(f"DEBUG: pdflatex run {run_num + 1} completed with return code: {result_proc.returncode}")
                        
                        if result_proc.returncode != 0:
                            print(f"DEBUG: pdflatex failed on run {run_num + 1}")
                            if run_num == 0:
                                break
                    
                    # Check if PDF was successfully created
                    if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 0:
                        print(f"DEBUG: PDF generated successfully! Size: {os.path.getsize(pdf_file)} bytes")
                        pdf_generated = True
                    else:
                        print(f"DEBUG: PDF file not found or empty after pdflatex")
                        
                else:
                    print(f"DEBUG: pdflatex not available, return code: {check_result.returncode}")
                    
            except subprocess.TimeoutExpired:
                print(f"DEBUG: pdflatex timed out during compilation")
            except Exception as e:
                print(f"DEBUG: Exception during PDF generation: {e}")
            
            if pdf_generated:
                # Return the PDF file directly
                with open(pdf_file, 'rb') as f:
                    pdf_content = f.read()
                
                print(f"DEBUG: Returning PDF ({len(pdf_content)} bytes)")
                return StreamingResponse(
                    io.BytesIO(pdf_content),
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=Braking_Report_{ts}.pdf"}
                )
            else:
                print(f"DEBUG: PDF generation failed, returning error")
                raise HTTPException(500, "PDF generation failed - pdflatex compilation error")
        
    except Exception as e:
        print(f"DEBUG: PDF DOWNLOAD EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
