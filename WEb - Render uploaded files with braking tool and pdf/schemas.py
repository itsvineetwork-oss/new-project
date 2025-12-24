from pydantic import BaseModel
from typing import Optional, Dict

# User jab Signup karega to yeh data bhejega
class UserCreate(BaseModel):
    email: str
    password: str

# User jab Login karega
class UserLogin(BaseModel):
    email: str
    password: str

# User jab License Activate karega
class LicenseActivate(BaseModel):
    license_key: str


class BrakingRawInput(BaseModel):
    mass_kg: float
    speed_kmh: float
    mu: Optional[float] = 0.3
    reaction_time: Optional[float] = 1.0
    gradient: Optional[float] = 0.0
    gradient_type: Optional[str] = 'percent'
    num_wheels: Optional[int] = 4
    max_braking_force: Optional[float] = None


class SimpleResponse(BaseModel):
    message: str
