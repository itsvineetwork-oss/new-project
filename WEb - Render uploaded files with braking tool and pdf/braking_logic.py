"""
Braking Performance Calculator - Backend Logic
Based on DIN EN 15746-2:2021-05 standards for railway vehicles
Matches the exact calculation logic from the desktop application (braking.py)
"""

import math
import re
from typing import Dict, Any, List

# =============================================================================
# CONSTANTS & STANDARDS
# =============================================================================
G = 9.81

# Standard braking distances (Reference for calculating Force Capability)
BRAKING_DATA = {
    8: 3, 10: 5, 16: 12, 20: 20, 24: 28,
    30: 45, 32: 50, 40: 75, 50: 135, 60: 180
}

# Limits for Compliance Checking
MAX_STOPPING_DISTANCES = {
    8: 6, 10: 9, 16: 18, 20: 27, 24: 36,
    30: 55, 32: 60, 40: 90, 50: 155, 60: 230,
    70: 300, 80: 400, 90: 500, 100: 620
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def parse_list(input_str):
    """Parse comma-separated string into list of floats"""
    if not input_str:
        return []
    try:
        # Handle "10, 20" or just "10"
        return [float(x.strip()) for x in str(input_str).split(',') if x.strip()]
    except:
        return []

def calculate_angle(gradient_val, gradient_type):
    """Convert gradient to angle in degrees"""
    if gradient_val == 0:
        return 0.0
    if gradient_type == "Degree (°)":
        return float(gradient_val)
    elif gradient_type == "1 in G":
        return math.degrees(math.atan(1 / gradient_val)) if gradient_val != 0 else 0
    else:  # Percentage
        return math.degrees(math.atan(gradient_val / 100))

def get_compliance(speed, total_dist):
    """Check if stopping distance complies with EN standard"""
    # Find the appropriate limit for the current speed
    allowed_distance = None
    for limit_speed in sorted(MAX_STOPPING_DISTANCES.keys(), reverse=True):
        if speed >= limit_speed:
            allowed_distance = MAX_STOPPING_DISTANCES[limit_speed]
            break
            
    if allowed_distance is None:
        return "Standard Not Found"
    
    if total_dist <= allowed_distance:
        return "✓ Standard Followed"
    else:
        return "✗ Standard Exceeded"

def escape_latex(s):
    """Escape special LaTeX characters"""
    if not isinstance(s, str):
        return s
    mapping = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}', ';': r'\;', ':': r'\:',
    }
    return re.sub(r'[&%$#_{}~^;:\\\\]', lambda m: mapping.get(m.group(0)), s)

# =============================================================================
# MAIN LOGIC
# =============================================================================
def perform_calculation_sequence(inputs):
    """
    Main calculation function matching desktop app logic exactly.
    Calculates braking performance for rail and road modes.
    """
    results_table_rows = []
    rail_detailed_calcs = []
    road_detailed_calcs = []
    
    mass_kg = inputs['mass_kg']
    weight_n = mass_kg * G
    reaction_time = inputs['reaction_time']
    num_wheels = inputs['num_wheels']
    
    # -------------------------------------------------------------------------
    # STEP 1: CALCULATE GLOBAL MAX BRAKING FORCE (RAIL)
    # -------------------------------------------------------------------------
    # This logic matches 'perform_calculations' in desktop app.
    # It checks ALL standard speeds to find the required force capability.
    
    old_data_for_report = {}
    max_rail_force = 0.0
    
    for speed, dist in sorted(BRAKING_DATA.items()):
        v_ms = speed / 3.6
        # Deceleration required on flat to meet this specific standard distance
        decel_required = (v_ms**2) / (2 * dist)
        force_required = mass_kg * decel_required
        
        # Store for PDF Table 1
        old_data_for_report[speed] = {
            'speed_ms': round(v_ms, 2),
            'braking_distance': dist,
            'deceleration': round(decel_required, 4),
            'reaction_distance': round(v_ms * reaction_time, 2),
            'total_stopping_distance': round((v_ms * reaction_time) + dist, 2),
            'braking_force': round(force_required, 2)
        }
        
        # Keep track of the maximum force required across all standards
        if force_required > max_rail_force:
            max_rail_force = force_required

    # -------------------------------------------------------------------------
    # STEP 2: RAIL CALCULATION LOOP (Using Global Max Force)
    # -------------------------------------------------------------------------
    rail_speeds = parse_list(inputs['rail_speed_input'])
    rail_gradients = parse_list(inputs['rail_gradient_input'])
    
    # Ensure 0 gradient is always calculated
    rail_gradients_with_flat = sorted(list(set([0.0] + rail_gradients)))
    
    for grad_val in rail_gradients_with_flat:
        
        # Determine Scenarios for PDF template compatibility
        scenarios = ["Straight Track"]
        if grad_val > 0:
            scenarios = ["Moving up", "Moving down"]  # Keep for PDF template
            
        for scenario in scenarios:
            for speed in sorted(rail_speeds):
                
                v_ms = speed / 3.6
                current_grad = 0 if scenario == "Straight Track" else grad_val
                angle_deg = calculate_angle(current_grad, inputs['rail_gradient_type'])
                
                grav_force_slope = weight_n * math.sin(math.radians(angle_deg))
                
                # Calculate net force exactly like original braking.py
                if scenario == "Straight Track":
                    f_net = max_rail_force
                    eff_grav = 0
                elif scenario == "Moving up":
                    f_net = max_rail_force + grav_force_slope  # Gravity helps
                    eff_grav = grav_force_slope
                elif scenario == "Moving down":
                    f_net = max_rail_force - grav_force_slope  # Gravity works against (original logic)
                    eff_grav = grav_force_slope
                
                # Calculate Deceleration exactly like original
                decel = abs(f_net / mass_kg) if mass_kg > 0 else 0.0
                
                # Calculate braking distance exactly like original
                if decel > 0 and v_ms > 0:
                    bd = abs((0 - v_ms**2) / (2 * decel))
                else:
                    bd = 0.0
                
                reaction_dist = v_ms * reaction_time
                total_dist = reaction_dist + bd
                
                # Check Compliance against Max Limits
                compliance = get_compliance(speed, total_dist)
                
                # 1. Add to Table Output
                results_table_rows.append({
                    "mode": "Rail",
                    "scenario": scenario,
                    "speed": speed,
                    "gradient": f"{current_grad} ({inputs['rail_gradient_type']})" if current_grad > 0 else "0",
                    "gradient_value": current_grad,
                    "gravitational_force": round(weight_n * math.sin(math.radians(angle_deg)), 2),
                    "applied_force": round(max_rail_force, 2),
                    "net_force": round(f_net, 2),
                    "decel": round(decel, 2),
                    "dist": round(bd, 2) if bd < 99999 else "Inf",
                    "total": round(total_dist, 2) if bd < 99999 else "Inf",
                    "status": compliance
                })
                
                # 2. Add to PDF List
                rail_detailed_calcs.append({
                    'scenario': scenario,
                    'speed_kmh': speed,
                    'v_ms': round(v_ms, 2),
                    'v_ms_squared': round(v_ms**2, 2),
                    'gradient_value': current_grad,
                    'angle_deg': round(angle_deg, 2),
                    'mass_kg': mass_kg,
                    'weight_n': round(weight_n, 2),
                    'fmax': round(weight_n * math.sin(math.radians(angle_deg)), 2),  # Holding Force
                    'f_g': round(eff_grav, 2),
                    'max_braking_force': round(max_rail_force, 2),
                    'f_net': round(f_net, 2),
                    'a_deceleration': round(decel, 2),
                    'a_deceleration_doubled': round(decel*2, 2),
                    'reaction_distance': round(reaction_dist, 2),
                    'braking_distance': round(bd, 2),
                    'total_stopping_distance': round(total_dist, 2)
                })

    # -------------------------------------------------------------------------
    # STEP 3: ROAD CALCULATION LOOP (Friction Based)
    # -------------------------------------------------------------------------
    if inputs['calc_mode'] == "Rail+Road":
        road_speeds = parse_list(inputs['road_speed_input'])
        road_gradients = parse_list(inputs['road_gradient_input'])
        road_gradients = sorted(list(set([0.0] + road_gradients)))
        
        for grad_val in road_gradients:
            for speed in sorted(road_speeds):
                
                v_ms = speed / 3.6
                angle = calculate_angle(grad_val, inputs['road_gradient_type'])
                
                normal = weight_n * math.cos(math.radians(angle))
                grav = weight_n * math.sin(math.radians(angle))
                
                # Friction Force
                friction_f = inputs['mu'] * normal
                
                # Restore scenarios for PDF template compatibility
                if grad_val == 0:
                    scenarios = [{"name": "Straight Track", "grav_factor": 0}]
                else:
                    scenarios = [
                        {"name": "Moving up", "grav_factor": 1},
                        {"name": "Moving down", "grav_factor": -1}
                    ]
                
                for scenario in scenarios:
                    # Calculate net force - simple physics
                    if scenario["name"] == "Straight Track":
                        net = friction_f
                    elif scenario["name"] == "Moving up":
                        net = friction_f + grav  # Friction + gravity help
                    else:  # Moving down
                        net = friction_f - grav  # Friction fights gravity (can be negative)
                    
                    # Calculate deceleration like original
                    decel = abs(net / mass_kg) if mass_kg > 0 else 0.0
                    
                    # Calculate braking distance like original
                    if decel > 0 and v_ms > 0:
                        bd = abs((0 - v_ms**2) / (2 * decel))
                    else:
                        bd = 0.0
                        
                    rd = v_ms * reaction_time
                    td = rd + bd
                    
                    gradient_display = f"{grad_val} ({inputs['road_gradient_type']})" if grad_val > 0 else "0"
                    
                    results_table_rows.append({
                        "mode": "Road",
                        "scenario": scenario["name"],
                        "speed": speed,
                        "gradient": gradient_display,
                        "gradient_value": grad_val,
                        "gravitational_force": round(grav, 2),
                        "applied_force": round(friction_f, 2),
                        "net_force": round(net, 2),
                        "decel": round(decel, 2),
                        "dist": round(bd, 2) if bd < 99999 else "Inf",
                        "total": round(td, 2) if td < 99999 else "Inf",
                        "status": "N/A"
                    })
                    
                    road_detailed_calcs.append({
                        'scenario': scenario["name"],
                    'gradient_value': grad_val, 'speed_kmh': speed,
                    'v_ms': round(v_ms, 2), 'v_ms_squared': round(v_ms**2, 2),
                    'mass_kg': mass_kg, 'weight_n': round(weight_n, 2),
                    'friction': inputs['mu'], 'angle_deg': round(angle, 2),
                    'fmax': round(grav, 2),
                    'normal_force': round(normal, 2), 'fb_friction': round(friction_f, 2),
                    'f_g': round(grav, 2), 'f_net': round(net, 2),
                    'a_deceleration': round(decel, 2), 'a_deceleration_doubled': round(decel*2, 2),
                    'reaction_distance': round(rd, 2), 'braking_distance': round(bd, 2),
                    'total_stopping_distance': round(td, 2)
                })

    # -------------------------------------------------------------------------
    # STEP 4: SUMMARY & CONTEXT
    # -------------------------------------------------------------------------
    # Find critical data for PDF template (Moving Down scenario)
    down_data = next((x for x in rail_detailed_calcs if x["scenario"] == "Moving down"), None)
    
    # Example calculation data (uses max speed input)
    max_input_speed = max(rail_speeds) if rail_speeds else 0
    ref_v = max_input_speed / 3.6
    
    # Find ref distance from table for this specific speed (or closest)
    ref_dist = 50
    for s in sorted(BRAKING_DATA.keys(), reverse=True):
        if max_input_speed >= s:
            ref_dist = BRAKING_DATA[s]
            break
            
    # Calculate values for the "Example" section of PDF
    ref_decel = (ref_v**2) / (2 * ref_dist)
    ref_force = mass_kg * ref_decel
    
    # Calculate GBR (Gross Braking Ratio)
    gbr = round((max_rail_force / (mass_kg * G)) * 100, 2) if mass_kg > 0 else 0

    # Create a "calc" object for template compatibility
    calc_obj = down_data if down_data else {}
    if not calc_obj:
        # Create default calc object if no down_data
        calc_obj = {
            'mass_kg': mass_kg,
            'weight_n': round(weight_n, 2),
            'speed_kmh': max_input_speed,
            'v_ms': round(ref_v, 2),
            'max_braking_force': round(max_rail_force, 2),
            'angle_deg': 0,
            'fmax': 0,
            'f_g': 0,
            'f_net': round(max_rail_force, 2),
            'a_deceleration': round(ref_decel, 2),
            'gradient_value': 0
        }

    context = {
        'doc_no': escape_latex(inputs.get('doc_no', '')),
        'made_by': escape_latex(inputs.get('made_by', '')),
        'checked_by': escape_latex(inputs.get('checked_by', '')),
        'approved_by': escape_latex(inputs.get('approved_by', '')),
        'mass_kg': mass_kg, 'weight_n': round(weight_n, 2),
        'speed_kmh': max_input_speed, 'v_ms': round(ref_v, 2),
        'reaction_time': reaction_time, 'Reaction_distance': round(ref_v * reaction_time, 2),
        'reference_speed_for_force': max_input_speed, 'reference_braking_dist': ref_dist,
        'decel': round(ref_decel, 2), 'totl_sto_distan': round((ref_v*reaction_time)+ref_dist, 2),
        'fb': round(ref_force, 2),
        'example_decel': round(ref_decel, 2), 'example_fb': round(ref_force, 2),
        'gradient_input': max(rail_gradients) if rail_gradients else 0,
        'gradient_type': inputs.get('rail_gradient_type', ''),
        'road_gradient_input': max(parse_list(inputs.get('road_gradient_input', '0'))) if inputs.get('road_gradient_input') else 0,
        'road_gradient_type': inputs.get('road_gradient_type', ''),
        'road_angle_deg': round(calculate_angle(max(parse_list(inputs.get('road_gradient_input', '0'))) if inputs.get('road_gradient_input') else 0, inputs.get('road_gradient_type', 'Percentage (%)')), 2),
        'number_of_wheels': num_wheels,
        'wheel_dia': inputs.get('wheel_dia', 0),
        'wheel_radius': inputs.get('wheel_dia', 0) / 2 if inputs.get('wheel_dia') else 0,
        'friction_coefficient': inputs.get('mu', 0.7),
        'max_braking_force': round(max_rail_force, 2),
        'min_braking_force': round(max_rail_force/num_wheels, 2) if num_wheels else 0,
        'old_data_for_report': old_data_for_report,
        'rail_detailed_calcs': rail_detailed_calcs,
        'road_detailed_calcs': road_detailed_calcs,
        'total_stopping_distance_ts_new__Moving_down': down_data['total_stopping_distance'] if down_data else 0,
        'fmax': down_data['fmax'] if down_data else 0,
        'gbr': gbr,
        'gbr_value': gbr / 100,  # GBR as decimal for template
        'gbr_percentage': gbr,   # GBR as percentage
        'speed_list': rail_speeds,
        # Add standard_speed_inputs for template compatibility
        'standard_speed_inputs': old_data_for_report,
        # Add calc object for template
        'calc': calc_obj,
        # Add checkbox controls for PDF report
        'show_gbr': inputs.get('show_gbr', False),
        'show_straight': inputs.get('show_straight', True),
        'show_moving_up': inputs.get('show_moving_up', True),
        'show_moving_down': inputs.get('show_moving_down', True)
    }
    
    return results_table_rows, context
