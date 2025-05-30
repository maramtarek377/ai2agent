import os
import json
import re
import logging
from datetime import date
from typing import TypedDict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId, SON
from pymongo import MongoClient
import requests
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and validate environment variables
def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"Missing {var_name} in environment")
    logger.info(f"Loaded {var_name}")
    return value

# Load environment variables with validation
try:
    logger.info("Loading environment variables...")
    MONGODB_URI = get_required_env("MONGODB_URI")
    GOOGLE_API_KEY = get_required_env("GOOGLE_API_KEY")
    MALE_BN_API_URL = get_required_env("MALE_BN_API_URL")
    FEMALE_BN_API_URL = get_required_env("FEMALE_BN_API_URL")
except EnvironmentError as e:
    logger.error(f"Environment configuration error: {str(e)}")
    raise

# Initialize MongoDB client
try:
    logger.info("Initializing MongoDB connection...")
    tmp_client = MongoClient(MONGODB_URI)
    db = tmp_client.get_default_database()
    patients_col = db["patients"]
    metrics_col = db["healthmetrics"]
    medications_col = db["medications"]
    medicines_col = db["medicines"]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Initialize LLM
try:
    logger.info("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Pydantic models
class Medication(BaseModel):
    medicationName: str
    dosage: str
    frequency: Optional[str] = None
    type: str = "continue" 
    rationale: Optional[str] = None

class Medicine(BaseModel):
    name: str
    specialization: str
    description: Optional[str] = None
    active_ingredient: Optional[str] = None
    dosage_forms: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    interactions: Optional[List[str]] = None

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[dict] = None
    exercise_plan: Optional[dict] = None
    nutrition_targets: Optional[dict] = None
    doctor_recommendations: Optional[List[str]] = None
    medication_recommendations: Optional[List[dict]] = None
    required_labs: Optional[List[dict]] = None

class State(TypedDict):
    patient_data: dict
    sent_for: int
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: List[str]
    current_medications: List[Medication]
    available_medicines: List[Medicine]

# Helper functions
def parse_probability(prob_str: str) -> float:
    return float(prob_str.strip('%')) / 100

def get_risk_probabilities(patient_data: dict) -> dict:
    payload = patient_data.copy()
    payload.pop('gender', None)
    gender = patient_data.get('gender')
    
    if gender == 'M':
        api_url = MALE_BN_API_URL
    elif gender == 'F':
        api_url = FEMALE_BN_API_URL
    else:
        raise ValueError("Invalid gender in patient data; must be 'M' or 'F'")

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"BN API request failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"BN service error: {str(e)}")

def classify_recommendation(text: str) -> str:
    t = text.lower()
    if 'exercise' in t:
        return 'Physical Activity'
    if 'diet' in t or 'nutrition' in t:
        return 'Diet'
    if 'smoking' in t:
        return 'Smoking Cessation'
    return 'Other'

def adjust_metrics(data: dict, kind: str) -> dict:
    d = data.copy()
    if kind == 'Physical Activity':
        d['Exercise_Hours_Per_Week'] = d.get('Exercise_Hours_Per_Week', 0) + 2
    if kind == 'Diet':
        if 'BMI' in d:
            d['BMI'] = max(d['BMI'] - 1, 0)
        if 'glucose' in d:
            d['glucose'] = max(d['glucose'] - 10, 0)
    if kind == 'Smoking Cessation':
        d['is_smoking'] = False
    return d

def is_effective(orig: dict, new: dict) -> bool:
    o = orig['Health Risk Probabilities']
    n = new['Health Risk Probabilities']
    o_d = parse_probability(o['Diabetes'])
    o_c = parse_probability(o['Heart Disease'])
    n_d = parse_probability(n['Diabetes'])
    n_c = parse_probability(n['Heart Disease'])
    return ((n_d < o_d - 0.05 and n_c <= o_c + 0.01) or
            (n_c < o_c - 0.05 and n_d <= o_d + 0.01))

def get_patient_medications(patient_id: str) -> List[Medication]:
    try:
        medications = list(medications_col.find({"patientId": patient_id}))
        return [
            Medication(
                medicationName=med.get('medicationName'),
                dosage=med.get('dosage'),
                frequency=med.get('frequency')
            ) for med in medications
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medications: {str(e)}")
        return []

def get_available_medicines() -> List[Medicine]:
    try:
        medicines = list(medicines_col.find({}))
        return [
            Medicine(
                name=med.get('name'),
                specialization=med.get('specialization'),
                description=med.get('description', ''),
                active_ingredient=med.get('active_ingredient', ''),
                dosage_forms=med.get('dosage_forms', []),
                contraindications=med.get('contraindications', []),
                interactions=med.get('interactions', [])
            ) for med in medicines
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medicines database: {str(e)}")
        return []

def build_patient_profile(input_values: dict) -> str:
    """Construct a detailed patient profile string from input values"""
    profile = []
   
    # Basic demographics
    if input_values.get('Age'):
        profile.append(f"Age: {input_values['Age']} years")
    if input_values.get('gender'):
        profile.append(f"Gender: {'Male' if input_values['gender'] == 'M' else 'Female'}")
   
    # Health metrics
    if input_values.get('BMI'):
        bmi = input_values['BMI']
        bmi_status = ""
        if bmi < 18.5: bmi_status = " (Underweight)"
        elif 18.5 <= bmi < 25: bmi_status = " (Normal)"
        elif 25 <= bmi < 30: bmi_status = " (Overweight)"
        else: bmi_status = " (Obese)"
        profile.append(f"BMI: {bmi}{bmi_status}")
   
    if input_values.get('Blood_Pressure'):
        print("Blood")
        bp = input_values['Blood_Pressure']
        bp_status = " (Normal)" if bp <= 120 else " (Elevated)" if bp <= 129 else " (Hypertension)"
        profile.append(f"Blood Pressure: {bp}{bp_status}")
    if 'hypertension' in input_values:
        profile.append("Blood Pressure: " + ("Hypertension" if input_values['hypertension'] else "Normal"))
   
    if input_values.get('glucose'):
        glucose = input_values['glucose']
        glucose_status = ""
        if glucose < 100: glucose_status = " (Normal)"
        elif 100 <= glucose < 126: glucose_status = " (Prediabetes)"
        else: glucose_status = " (Diabetes)"
        profile.append(f"Glucose: {glucose}{glucose_status}")
   
    # Lifestyle factors
    if input_values.get('Diet'):
        diet = input_values['Diet'].lower()
        if diet == 'healthy': diet_desc = "Healthy diet"
        elif diet == 'unhealthy': diet_desc = "Unhealthy diet"
        else: diet_desc = "Average diet"
        profile.append(diet_desc)
   
    if input_values.get('Exercise_Hours_Per_Week'):
        exercise = input_values['Exercise_Hours_Per_Week']
        if exercise == 0: ex_desc = "Sedentary (no exercise)"
        elif exercise < 3: ex_desc = "Light activity"
        elif exercise < 5: ex_desc = "Moderate activity"
        else: ex_desc = "Active"
        profile.append(f"Exercise: {ex_desc} ({exercise} hrs/week)")
   
    if input_values.get('Stress_Level'):
        stress = input_values['Stress_Level']
        if stress < 4: stress_desc = "Low stress"
        elif stress < 7: stress_desc = "Moderate stress"
        else: stress_desc = "High stress"
        profile.append(f"Stress Level: {stress}/10 ({stress_desc})")
   
    # Risk factors
    if input_values.get('is_smoking') is not None:
        profile.append("Smoker" if input_values['is_smoking'] else "Non-smoker")
   
   
    if input_values.get('is_alcohol_user') is not None:
        profile.append("Alcohol Use: " + ("Yes" if input_values['is_alcohol_user'] else "No"))
    if input_values.get('CVD_Family_History') is not None:
        profile.append("Family History of CVD: " + ("Yes" if input_values['CVD_Family_History'] else "No"))
   
    if input_values.get('admission_tsh') is not None:
        tsh = input_values['admission_tsh']
        tsh_status = "N/A" if tsh == 0 else" (Normal)" if 0.4 <= tsh <= 4.0 else " (Low)" if tsh < 0.4 else " (High)"
        profile.append(f"TSH: {tsh} mIU/L{tsh_status}")
    if input_values.get('creatine_kinase_ck') is not None:
        ck = input_values['creatine_kinase_ck']
        ck_status = "N/A" if ck == 0 else" (Normal)" if ck < 200 else " (Elevated)"
        profile.append(f"Creatine Kinase: {ck} U/L{ck_status}")
    if input_values.get('ld_value') is not None:
       
        ldl = input_values['ld_value']
        ldl_status = " (N/A)" if ldl==0 else" (Optimal)" if ldl > 140  and ldl < 280 else " (Not Optimal)"
        profile.append(f"LDh : {ldl} mg/dL{ldl_status}")
    if input_values.get('hemoglobin_a1c') is not None:
        a1c = input_values['hemoglobin_a1c']
        a1c_status = "N/A" if a1c == 0 else" (Normal)" if a1c < 5.7 else " (Prediabetes)" if a1c < 6.5 else " (Diabetes)"
        profile.append(f"Hemoglobin A1c: {a1c}%{a1c_status}")
    if input_values.get('Sleep_Hours_Per_Day') is not None:
        sleep = input_values['Sleep_Hours_Per_Day']
        sleep_status = " (Insufficient)" if sleep < 7 else " (Adequate)" if sleep <= 9 else " (Excessive)"
        profile.append(f"Sleep: {sleep} hours/day{sleep_status}")

    if input_values.get('Diabetes_pedigree') is not None :
        profile.append("Family History of Diabetes: " + ("Yes" if input_values['Diabetes_pedigree'] else "No"))

    # Regional info
    profile.append("Region: Egypt")
 
   
    return "\n".join(profile)

def generate_patient_prompt(input_values: dict, risk_probs: dict, medications: List[Medication]) -> str:
    """Generate dynamic prompt for patient recommendations with personalized diet"""
    patient_profile = build_patient_profile( input_values)
   
    # Determine diet requirements based on patient conditions
    diet_requirements = []
   
    # Gender-based requirements
    gender = input_values.get('gender', '')
    if gender == 'M':
        diet_requirements.append("higher protein needs")
    elif gender == 'F':
        diet_requirements.append("adequate iron and calcium")
   
    # Age-based requirements
    age = input_values.get('Age', 0)
    if age > 50:
        diet_requirements.append("higher fiber and calcium")
    if age > 65:
        diet_requirements.append("easier to digest foods")
   
    # Condition-based requirements
    if input_values.get('hypertension'):
        diet_requirements.append("low sodium (<1500mg/day)")
   
    diabetes_risk = parse_probability(risk_probs['Diabetes'])
    if diabetes_risk > 0.25:  # >25% risk
        diet_requirements.append("low glycemic index foods")
        if diabetes_risk > 0.5:  # >50% risk
            diet_requirements.append("controlled carbohydrate intake")
   
    cvd_risk = parse_probability(risk_probs['Heart Disease'])
    if cvd_risk > 0.2:  # >20% risk
        diet_requirements.append("heart-healthy fats")
        diet_requirements.append("low saturated fat")
   
    bmi = input_values.get('BMI', 0)
    if bmi >= 30:
        diet_requirements.append("calorie-controlled for weight loss")
    elif bmi >= 25:
        diet_requirements.append("moderate calorie reduction")
       # Consider additional factors from patient data
    if input_values.get('is_smoking'):
        diet_requirements.append("antioxidant-rich foods")
   
    if input_values.get('Stress_Level', 0) > 6:
        diet_requirements.append("stress-reducing nutrients (magnesium, omega-3s)")
   
    if input_values.get('Sleep_Hours_Per_Day', 0) < 6:
        diet_requirements.append("sleep-promoting foods")
   
    # Build diet focus description
    diet_focus = f"Focus on: {', '.join(diet_requirements)}" if diet_requirements else "Balanced nutrition"
   
    # Get exercise and stress details for exercise plan
    exercise_per_week = input_values.get('Exercise_Hours_Per_Week', 0)
    stress_level = input_values.get('Stress_Level', 0)
    target_bmi = max(input_values.get('BMI', 0) - 1, 18.5)  # Aim for healthy BMI
   
    # Build prompt parts separately to avoid deep nesting
    prompt_parts = [
        "Generate a fully personalized health and nutrition plan for a patient using the profile below:",
        "",
        f"Patient Profile:",
        patient_profile,
        "",
        "Include the following sections, returning only a JSON object:",
        "",
        "1. patient_recommendations:  // List of concise, actionable tips tailored to this individual's conditions, medications, lifestyle, and priorities",
        "   [",
        '     "Advice 1",',
        '     "Advice 2",',
        '     "..."',
        "   ]",
        "",
        "2. diet_plan:",
        "   {",
        '     "description": "Brief overview of the plan and how it meets goals",',
        '     "daily_calories": {',
        '       "target": "kcal based on gender, age, BMI, activity level",',
        '       "range": "min-max kcal"',
        "     },",
        '     "macronutrients": {',
        '       "carbohydrates": {',
        '         "grams": "g (X% of total)",',
        '         "focus": "adjusted for diabetes risk"',
        "       },",
        '       "protein": {',
        '         "grams": "g",',
        '         "focus": "gender- and age-specific needs"',
        "       },",
        '       "fats": {',
        '         "grams": "g",',
        '         "type": "e.g., MUFA, PUFA",',
        '         "focus": "heart health and CVD risk"',
        "       }",
        "     },",
        f'     "key_focus": {json.dumps(diet_requirements)},',
        '     "meals": [  // provide 5 days of varied, culturally diverse, portioned meals in grams',
        "       {",
        '         "day": 1,',
        '         "breakfast": {"item": "...", "grams":  ...},',
        '         "lunch":     {"item": "...", "grams":  ...},',
        '         "dinner":    {"item": "...", "grams":  ...},',
        '         "snacks":    [{"item": "...", "grams": ...}, ...]',
        "       },",
        "       // days 2-5, including at least some non-Egyptian dishes for variety",
        "     ],",
        '     "avoid": ["...", "..."],',
        '     "hydration": "liters per day"',
        "   }",
        "",
        "3. exercise_plan:",
        "   {",
        '     "weekly_schedule": [',
        '       {',
        f'         "day": "Monday",',
        f'         "activity": "Personalized based on age={input_values.get("Age", "N/A")}, BMI={input_values.get("BMI", "N/A")}",',
        f'         "duration_min": "Based on current fitness level: {exercise_per_week} hrs/week",',
        f'         "intensity": "Adjusted for health risks: Diabetes={risk_probs["Diabetes"]}, CVD={risk_probs["Heart Disease"]}"',
        '       },',
        '       // Include all 7 days with personalized activities',
        '       // Include at least 2 rest days',
        '       // Vary intensity based on stress levels and health conditions',
        '     ],',
        f'     "type_recommendations": "Mix of aerobic, strength, and flexibility exercises personalized by age={input_values.get("Age", "N/A")}, gender={input_values.get("gender", "N/A")}, health risks={risk_probs}, and target BMI={target_bmi}",',
        '     "frequency_recommendations": {',
        f'       "current_weekly_exercise": "{exercise_per_week} sessions/week",',
        '       "suggestion": "If current frequency is below the target level for her/his profile, provide a progressive plan to increase frequency safely; if adequate, maintain or optimize intensity/duration."',
        '     },',
        f'     "stress_management": " if value of stress is high only (stress={input_values.get("Stress_Level", "N/A")} out of 10)Include low-impact, mindfulness-integrated activities (e.g., yoga, tai chi) on high-stress days based on stress_level={stress_level}",',
        f'     "hypertension_precautions": "Low-impact cardio and resistance training with monitored intensity to manage blood pressure; avoid high-intensity intervals if uncontrolled all is related to the patient blood pressure({input_values.get("Blood_Pressure", "N/A")})",',
        '     "progression": "Detailed 4-week progression plan to increase intensity or volume as tolerated if needed",',
        '     "precautions": "Any special precautions based on health conditions and medication interactions"',
        "   }",
        "",
        "4. nutrition_targets:",
        "   {",
        f'     "target_BMI": "{target_bmi} by {date.today().replace(year=date.today().year + 1).strftime("%Y-%m-%d")}",',
        f'     "target_glucose": "mg/dL range based on diabetes risk ={risk_probs["Diabetes"]}, ",',
        '     "other": {',
        '       "blood_pressure": "Target based on current status",',
        f'       "cholesterol": "Target levels based on CVD risk CVD={risk_probs["Heart Disease"]}""',
        '     }',
        "   }",
        "",
        "Requirements:",
        "- Use all inputs: age, gender, BMI, activity, exercise per week, stress level, sleep, current meds, health risks",
        "- Focus on macronutrient grams and percentages",
        "- Provide culturally sensitive choices with at least 30% non-Egyptian dishes",
        "- Ensure variety: rotate cuisines and ingredients daily",
        "- Adjust sodium, fiber, glycemic index per condition",
        "- Return ONLY the JSON object—no extra text",
        ""
    ]
   
    return "\n".join(prompt_parts)

def generate_doctor_prompt(input_values: dict, risk_probs: dict, medications: List[Medication], 
                         available_meds: List[Medicine], specialty: str) -> str:
    """Generate dynamic prompt for specialist recommendations with detailed medication analysis"""
    patient_profile = build_patient_profile( input_values)
    
    # Filter medicines by specialty and enforce strict specialty boundaries
    specialty_lower = specialty.lower()
    specialty_meds = []
    
    # Define strict specialty boundaries for medications
    if specialty_lower == "cardiology":
        specialty_meds = [m for m in available_meds 
                         if any(s.lower() in ['cardiac', 'cardiovascular', 'hypertension']) 
                         for s in m.specialization.split(',')]
        prohibited_conditions = ['diabetes', 'endocrine']
    elif specialty_lower == "endocrinology":
        specialty_meds = [m for m in available_meds 
                          if any(s.lower() in ['diabetes', 'endocrine', 'metabolic'] 
                          for s in m.specialization.split(','))]
        prohibited_conditions = ['cardiac', 'cardiovascular', 'hypertension']
    else:
        specialty_meds = []
        prohibited_conditions = []
    
    # Analyze current medications for potential issues
    current_meds_str = ""
    if medications:
        current_meds_str = "Current Medications:\n"
        for med in medications:
            current_meds_str += f"- {med.medicationName} ({med.dosage}, {med.frequency})\n"
    
    # Generate medication recommendations based strictly on specialty
    medication_needs = []
    
    # Cardiology-specific medication needs - only cardiac-related conditions
    if specialty_lower == "cardiology":
        hypertension_status = input_values.get('hypertension', False)
        blood_pressure = input_values.get('Blood_Pressure', 120)
        
        if hypertension_status or blood_pressure > 130:
            medication_needs.append({
                "condition": "Hypertension",
                "priority": "High" if blood_pressure > 140 else "Moderate",
                "allowed_types": ["ACE inhibitors", "ARBs", "beta-blockers", "calcium channel blockers", "diuretics"]
            })
            
        if parse_probability(risk_probs['Heart Disease']) > 0.2:
            medication_needs.append({
                "condition": "Cardiovascular Disease Prevention",
                "priority": "High" if parse_probability(risk_probs['Heart Disease']) > 0.3 else "Moderate",
                "allowed_types": ["statins", "antiplatelets"]
            })
            
        if input_values.get('ld_value', 0) > 130:
            medication_needs.append({
                "condition": "Hyperlipidemia",
                "priority": "High" if input_values['ld_value'] > 160 else "Moderate",
                "allowed_types": ["statins", "fibrates", "PCSK9 inhibitors"]
            })
    
    # Endocrinology-specific medication needs - only endocrine-related conditions
    elif specialty_lower == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            medication_needs.append({
                "condition": "Diabetes Prevention",
                "priority": "High" if parse_probability(risk_probs['Diabetes']) > 0.5 else "Moderate",
                "allowed_types": ["metformin", "SGLT2 inhibitors", "GLP-1 receptor agonists"]
            })
            
        if input_values.get('BMI', 0) > 30:
            medication_needs.append({
                "condition": "Obesity Management",
                "priority": "High" if input_values['BMI'] > 35 else "Moderate",
                "allowed_types": ["GLP-1 receptor agonists", "orlistat"]
            })
            
        if input_values.get('hemoglobin_a1c', 0) > 6.5:
            medication_needs.append({
                "condition": "Diabetes Treatment",
                "priority": "High",
                "allowed_types": ["insulin", "combination therapies"]
            })
    
    # Generate lab recommendations based on patient status
    lab_recommendations = []
    
    if specialty_lower == "cardiology":
        hypertension_status = input_values.get('hypertension', False)
        blood_pressure = input_values.get('Blood_Pressure', 120)
        
        if hypertension_status or blood_pressure > 130:
            lab_recommendations.append({
                "test": "Basic metabolic panel",
                "frequency": "Annually",
                "rationale": "Monitor electrolytes and kidney function in hypertensive patients"
            })
            
            if blood_pressure > 140:
                lab_recommendations.append({
                    "test": "Urinalysis",
                    "frequency": "Now and annually",
                    "rationale": "Assess for proteinuria in uncontrolled hypertension"
                })
        
        if parse_probability(risk_probs['Heart Disease']) > 0.2:
            lab_recommendations.append({
                "test": "Lipid panel",
                "frequency": "Every 6 months",
                "rationale": "Monitor LDL in high CVD risk patient"
            })
            
            if input_values.get('is_smoking', False):
                lab_recommendations.append({
                    "test": "High-sensitivity CRP",
                    "frequency": "Annually",
                    "rationale": "Assess inflammatory markers in smoker with high CVD risk"
                })
    
    elif specialty_lower == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            lab_recommendations.append({
                "test": "Hemoglobin A1c",
                "frequency": "Every 3 months",
                "rationale": "Monitor glycemic control in high diabetes risk"
            })
            
            if input_values.get('BMI', 0) > 30:
                lab_recommendations.append({
                    "test": "Fasting insulin",
                    "frequency": "Now",
                    "rationale": "Assess insulin resistance in obese patient"
                })
        
        if input_values.get('admission_tsh'):
            lab_recommendations.append({
                "test": "Free T4",
                "frequency": "Now",
                "rationale": "Complete thyroid function assessment"
            })
    
    # Build strict medication rules section
    medication_rules = [
        "STRICT MEDICATION RULES:",
        f"1. As a {specialty} specialist, you MUST ONLY recommend medications for:",
        f"   - {specialty}-related conditions",
        "2. You MUST NOT recommend medications for:",
        f"   - {', '.join(prohibited_conditions)} conditions" if prohibited_conditions else "   - No restrictions",
        "3. Medication recommendations must be:",
        "   - Within your specialty scope",
        "   - Supported by patient's conditions and risk factors",
        "   - Compatible with current medications",
        "",
        "Violating these rules will result in invalid recommendations."
    ]
    
    # Build medication recommendation section
    medication_section = []
    if specialty_meds:
        medication_section.append("Approved Specialty Medications in Database:")
        for med in specialty_meds[:5]:  # Limit to top 5 to avoid overwhelming
            med_info = f"- {med.name}: {med.specialization}. "
            if med.dosage_forms:
                med_info += f"Forms: {', '.join(med.dosage_forms[:2])}. "
            medication_section.append(med_info)
    else:
        medication_section.append("No specialty-specific medications found in database")
    
    medication_section.append("\nApproved Medication Needs Based on Patient Condition:")
    for need in medication_needs:
        medication_section.append(f"- {need['condition']} ({need['priority']} priority)")
        medication_section.append(f"  Allowed medication classes: {', '.join(need['allowed_types'])}")
    
    prompt_parts = [
        f"Generate comprehensive {specialty} recommendations for this patient:",
        patient_profile,
        "",
        "Health Risks:",
        f"- Diabetes: {risk_probs['Diabetes']}",
        f"- Heart Disease: {risk_probs['Heart Disease']}",
        "",
        current_meds_str,
        "",
        "\n".join(medication_rules),
        "",
        "\n".join(medication_section),
        "",
        "Provide recommendations in this exact JSON format:",
        "{",
        '    "doctor_recommendations": [',
        '        "Key clinical findings and prioritized risk factors",',
        '        "Summary of medication recommendations (SPECIALTY-APPROPRIATE ONLY)",',
        '        "Critical questions to ask patient",',
        '        "Red flags requiring immediate attention"',
        '    ],',
        '    "medication_recommendations": [',
        '        {',
        '            "name": "SPECIALTY-APPROPRIATE medication only",',
        '            "type": "new/adjustment/discontinuation",',
        '            "current_status": "new/current",', 
        '            "dosage": "specific dosage with units",',
        '            "frequency": "how often",',
        '            "duration": "how long to take",',
        '            "rationale": "why this medication is recommended",',
        '            "source": "database/general",',
        '            "specialty_appropriate": true',
        '        }',
        '    ],',
        '    "required_labs": [',
        '        {',
        '            "test_name": "name",',
        '            "frequency": "how often",',
        '            "rationale": "clinical reason",',
        '            "urgency": "when to perform"',
        '        }',
        '    ],',
        '    "diet_plan": {',
        '        "description": "brief dietary recommendations specific to condition",',
        '        "key_focus": ["list", "of", "dietary", "priorities"],',
        '        "avoid": ["list", "of", "foods", "to", "avoid"]',
        '    }',
        "}",
        "",
        "SPECIALTY-SPECIFIC GUIDELINES:",
        f"1. As a {specialty} specialist:",
        "   - Focus ONLY on conditions within your specialty",
        "   - Do NOT make recommendations outside your scope",
        "   - Flag any concerning findings from other specialties",
        "2. For medications:",
        "   - First check approved specialty medications list",
        "   - If no match, recommend GENERAL CLASS (not specific drug)",
        "   - ALWAYS verify medication is specialty-appropriate",
        "   - make sure that suggested medication not conflict with current medication and has positive effect",
        "   - make sure if they conflict tell which has periorty to take and time needed after finish first to stay to take the second due to The active  ingredient in medicine time take to leave body ok ",
        "3. For labs:",
        "   - Recommend only tests relevant to your specialty also depend on patient data",
        "   - Include clinical rationale for each test",
        "4. For current medications:",
        "   - Evaluate each current medication for:",
        "     * Continuation (if beneficial and no conflicts)",
        "     * Discontinuation (if harmful or conflicting)",
        "   - For each current medication recommendation:",
        '     * Set "type": "continue" or "stop"',
        "     * Provide rationale for decision",
        "     * Specify any dosage adjustments if continuing",
        "",
        "Return ONLY the JSON object with no additional text or explanations."
    ]
    
    return "\n".join(prompt_parts)

# Graph nodes
def risk_assessment(state: State) -> dict:
    probs = get_risk_probabilities(state['patient_data'])
    return {'risk_probabilities': probs}

def generate_recommendations(state: State) -> dict:
    input_values = state['risk_probabilities']['Input Values']
    risk_probs = state['risk_probabilities']['Health Risk Probabilities']
    sent_for = state['sent_for']
    medications = state.get('current_medications', [])
    available_meds = state.get('available_medicines', [])
    
    try:
        if sent_for == 0:  # General patient recommendations
            prompt = generate_patient_prompt(input_values, risk_probs, medications)
        elif sent_for == 1:  # Cardiology
            prompt = generate_doctor_prompt(input_values, risk_probs, medications, available_meds, "Cardiology")
        elif sent_for == 2:  # Endocrinology
            prompt = generate_doctor_prompt(input_values, risk_probs, medications, available_meds, "Endocrinology")
        else:
            raise HTTPException(status_code=400, detail='Invalid sent_for value')
        
        logger.info(f"Generated prompt:\n{prompt}")
        resp = llm.invoke(prompt)
        json_str = re.search(r'\{.*\}', resp.content, re.DOTALL).group(0)
        data = json.loads(json_str)
        
        # Process doctor recommendations if present
        if 'doctor_recommendations' in data and data['doctor_recommendations']:
            processed_recs = []
            for rec in data['doctor_recommendations']:
                if isinstance(rec, dict):
                    key = next(iter(rec))
                    processed_recs.append(f"{key}: {rec[key]}")
                else:
                    processed_recs.append(rec)
            data['doctor_recommendations'] = processed_recs
        
        # Process medication recommendations
        if 'medication_recommendations' in data:
            updated_medications = []
            
            # First process current medications
            for med in medications:
                # Default to continue unless specified otherwise
                med.type = "continue"
                updated_medications.append(med)
            
            # Now apply recommendations
            for med_rec in data['medication_recommendations']:
                if med_rec.get('current_status') == 'current':
                    # Find and update existing medication
                    for med in updated_medications:
                        if med.medicationName.lower() == med_rec['name'].lower():
                            # Set type based on recommendation
                            med.type = 'continue' if med_rec['type'] in ['continue', 'adjustment'] else 'stop'
                            
                            # Update dosage and frequency if adjustment
                            if med_rec['type'] == 'adjustment':
                                med.dosage = med_rec.get('dosage', med.dosage)
                                med.frequency = med_rec.get('frequency', med.frequency)
                            
                            # Add rationale if available
                            if 'rationale' in med_rec:
                                med.rationale = med_rec['rationale']
                            break
                elif med_rec.get('type') == 'new':
                    # Add new medications to the list
                    new_med = Medication(
                        medicationName=med_rec['name'],
                        dosage=med_rec.get('dosage', ''),
                        frequency=med_rec.get('frequency', ''),
                        type='new'
                    )
                    if 'rationale' in med_rec:
                        new_med.rationale = med_rec['rationale']
                    updated_medications.append(new_med)
            
            # Update state with processed medications
            state['current_medications'] = updated_medications
        
        return {'recommendations': Recommendations(**data)}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse recommendation response")
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

def evaluate_recommendations(state: State) -> dict:
    if state['sent_for'] != 0:
        return {'selected_patient_recommendations': []}
    
    original = state['risk_probabilities']
    selected = []
    for rec in state['recommendations'].patient_recommendations or []:
        kind = classify_recommendation(rec)
        if kind != 'Other':
            adj = adjust_metrics(state['patient_data'], kind)
            new_probs = get_risk_probabilities(adj)
            if is_effective(original, new_probs):
                selected.append(rec)
    return {'selected_patient_recommendations': selected}

def output_results(state: State) -> dict:
    probs = state['risk_probabilities']['Health Risk Probabilities']
    result = {
        'diabetes_probability': probs['Diabetes'],
        'cvd_probability': probs['Heart Disease'],
        'current_medications': [{
            'medicationName': m.medicationName,
            'dosage': m.dosage,
            'frequency': m.frequency,
            'type': m.type 
        } for m in state.get('current_medications', [])]
    }
    
    
    if state['sent_for'] == 0:
        result.update({
            'patient_recommendations': state['selected_patient_recommendations'][:3],
            'diet_plan': state['recommendations'].diet_plan,
            'exercise_plan': state['recommendations'].exercise_plan,
            'nutrition_targets': state['recommendations'].nutrition_targets
        })
    else:
        # Include all doctor-specific outputs
        rec_data = state['recommendations'].dict()
        result.update({
            'doctor_recommendations': rec_data.get('doctor_recommendations', [])[:6],
            'medication_recommendations': rec_data.get('medication_recommendations', [])[:3],
            'required_labs': rec_data.get('required_labs', [])[:3],
            'diet_plan': rec_data.get('diet_plan', {})
        })
    
    return result

# Build and compile state graph
graph_builder = StateGraph(State)
for node in ['risk_assessment', 'generate_recommendations', 'evaluate_recommendations', 'output_results']:
    graph_builder.add_node(node, globals()[node])

graph_builder.add_edge(START, 'risk_assessment')
graph_builder.add_edge('risk_assessment', 'generate_recommendations')
graph_builder.add_edge('generate_recommendations', 'evaluate_recommendations')
graph_builder.add_edge('evaluate_recommendations', 'output_results')
graph_builder.add_edge('output_results', END)

graph = graph_builder.compile()

# FastAPI app
app = FastAPI()

@app.get("/recommendations/{patient_id}")
async def get_recommendations(patient_id: str, sent_for: Optional[int] = 0):
    try:
        oid = ObjectId(patient_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    patient = patients_col.find_one({"_id": oid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    metrics = list(metrics_col.find({"patientId": patient_id}).sort([('createdAt', -1)]).limit(1))
    if metrics:
        patient.update(metrics[0])

    # Get patient medications
    medications = get_patient_medications(patient_id)
    
    # Get available medicines from database
    available_medicines = get_available_medicines()

    # Build patient data with defaults for required fields
    patient_data = {
        "Blood_Pressure": patient.get('bloodPressure', 120),  # Default to normal if not provided
        "Age": patient.get('anchorAge', 30),  # Default age if not provided
        "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek', 0),
        "Diet": patient.get('diet', 'average'),
        "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay', 7),
        "Stress_Level": patient.get('stressLevel', 5),
        "glucose": patient.get('glucose', 90),
        "BMI": patient.get('bmi', 22),
        "hypertension": patient.get('bloodPressure', 120) > 130,  # Determine from BP if not provided
        "is_smoking": patient.get('isSmoker', False),
        "hemoglobin_a1c": patient.get('hemoglobinA1c', 5.5),
        "Diabetes_pedigree": patient.get('diabetesPedigree', 0.5),
        "CVD_Family_History": patient.get('ckdFamilyHistory', False),
        "ld_value": patient.get('cholesterolLDL', 100),
        "admission_tsh": patient.get('admissionSOH', 2.5),
        "is_alcohol_user": patient.get('isAlcoholUser', False),
        "creatine_kinase_ck": patient.get('creatineKinaseCK', 100),
        "gender": 'M' if patient['gender'].lower().startswith('m') else 'F',
    }

    initial_state = {
        'patient_data': patient_data,
        'sent_for': sent_for,
        'current_medications': medications,
        'available_medicines': available_medicines
    } 
    result = await graph.ainvoke(initial_state)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
