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
        bp = input_values['Blood_Pressure']
        bp_status = " (Normal)" if bp <= 120 else " (Elevated)" if bp <= 129 else " (Hypertension)"
        profile.append(f"Blood Pressure: {bp}{bp_status}")
    
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
    if input_values.get('is_smoking'):
        profile.append("Smoker" if input_values['is_smoking'] else "Non-smoker")
    
    if input_values.get('hypertension'):
        profile.append("Hypertension" if input_values['hypertension'] else "Normal blood pressure")
    
    # Regional info
    profile.append("Region: Egypt")
    
    return "\n".join(profile)

def generate_patient_prompt(input_values: dict, risk_probs: dict, medications: List[Medication]) -> str:
    """Generate dynamic prompt for patient recommendations with personalized diet"""
    patient_profile = build_patient_profile(input_values)
    
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
        f'     "stress_management": "Include low-impact, mindfulness-integrated activities (e.g., yoga, tai chi) on high-stress days based on stress_level={stress_level}",',
        '     "hypertension_precautions": "Low-impact cardio and resistance training with monitored intensity to manage blood pressure; avoid high-intensity intervals if uncontrolled",',
        '     "progression": "Detailed 4-week progression plan to increase intensity or volume as tolerated",',
        '     "precautions": "Any special precautions based on health conditions and medication interactions"',
        "   }",
        "",
        "4. nutrition_targets:",
        "   {",
        f'     "target_BMI": "{target_bmi} by {date.today().replace(year=date.today().year + 1).strftime("%Y-%m-%d")}",',
        '     "target_glucose": "mg/dL range based on diabetes risk",',
        '     "other": {',
        '       "blood_pressure": "Target based on current status",',
        '       "cholesterol": "Target levels based on CVD risk"',
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
    """Generate dynamic prompt for specialist recommendations with detailed structured output"""
    patient_profile = build_patient_profile(input_values)
    
    # Filter medicines by specialty
    specialty_meds = [m for m in available_meds if specialty.lower() in m.specialization.lower()]
    
    # Current medications string
    current_meds_str = ""
    if medications:
        current_meds_str = "Current Medications:\n"
        for med in medications:
            current_meds_str += f"- {med.medicationName} ({med.dosage}, {med.frequency})\n"
    
    # Generate dynamic questions based on patient inputs
    questions_to_ask = []
    
    # Cardiology-specific questions
    if specialty.lower() == "cardiology":
        if input_values.get('hypertension'):
            questions_to_ask.append("Duration and severity of hypertension symptoms")
            questions_to_ask.append("Any episodes of chest pain or palpitations")
        if input_values.get('is_smoking'):
            questions_to_ask.append("Smoking history (pack-years)")
        if input_values.get('CVD_Family_History'):
            questions_to_ask.append("Details of family history of cardiovascular disease")
        if input_values.get('Blood_Pressure', 0) > 140:
            questions_to_ask.append("Any symptoms of hypertensive urgency (headache, visual changes)")
        if input_values.get('ld_value', 0) > 130:
            questions_to_ask.append("Previous attempts at cholesterol management")
    
    # Endocrinology-specific questions
    elif specialty.lower() == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            questions_to_ask.append("Symptoms of hyperglycemia (polyuria, polydipsia)")
            questions_to_ask.append("History of hypoglycemic episodes")
        if input_values.get('BMI', 0) > 30:
            questions_to_ask.append("Weight history and previous weight loss attempts")
        if input_values.get('hemoglobin_a1c', 0) > 5.7:
            questions_to_ask.append("Dietary habits and carbohydrate intake patterns")
        if input_values.get('admission_tsh'):
            questions_to_ask.append("Symptoms of thyroid dysfunction (fatigue, weight changes)")
    
    # General questions based on inputs
    if input_values.get('Stress_Level', 0) > 6:
        questions_to_ask.append("Stress management techniques currently used")
    if input_values.get('Exercise_Hours_Per_Week', 0) < 2:
        questions_to_ask.append("Barriers to physical activity")
    if input_values.get('Sleep_Hours_Per_Day', 0) < 6:
        questions_to_ask.append("Sleep quality and any sleep disturbances")
    
    # Build the structured prompt
    prompt_parts = [
        f"Generate comprehensive {specialty} recommendations for this patient in a structured format:",
        "",
        "Patient Profile:",
        patient_profile,
        "",
        "Health Risk Probabilities:",
        f"- Diabetes: {risk_probs['Diabetes']}",
        f"- Heart Disease: {risk_probs['Heart Disease']}",
        "",
        current_meds_str,
        "",
        f"Available {specialty} Medications:",
        "\n".join([f"- {m.name} ({m.active_ingredient})" for m in specialty_meds]) if specialty_meds else "None",
        "",
        "Return recommendations in this exact structured JSON format:",
        "{",
        '    "clinical_assessment": {',
        '        "key_findings": ["list of 3-5 most clinically relevant findings based on patient profile"],',
        '        "risk_factors": ["prioritized list of modifiable risk factors"],',
        '        "prognosis": "brief assessment of likely outcomes if no changes are made"',
        '    },',
        '    "medication_plan": {',
        '        "current_medication_review": [',
        '            {',
        '                "medication": "name",',
        '                "assessment": "appropriateness for this patient",',
        '                "recommendation": "continue/adjust/discontinue with rationale"',
        '            }',
        '        ],',
        '        "new_medications": [',
        '            {',
        '                "name": "medication name from available list",',
        '                "dosage": "specific dosage in mg",',
        '                "frequency": "times per day/week",',
        '                "duration": "how long to take",',
        '                "rationale": "why this is being prescribed",',
        '                "monitoring": "what to monitor and how often"',
        '            }',
        '        ],',
        '        "interaction_warnings": [',
        '            {',
        '                "medication_pair": "med1 + med2",',
        '                "interaction": "nature of interaction",',
        '                "management": "how to mitigate risk"',
        '            }',
        '        ]',
        '    },',
        '    "monitoring_plan": {',
        '        "vital_signs": {',
        '            "frequency": "how often to check",',
        '            "targets": {',
        '                "blood_pressure": "target range",',
        '                "heart_rate": "target range",',
        '                "weight": "target frequency and goal"',
        '            }',
        '        },',
        '        "laboratory_tests": [',
        '            {',
        '                "test": "test name",',
        '                "frequency": "how often",',
        '                "rationale": "clinical reason",',
        '                "urgency": "immediately/soon/routine"',
        '            }',
        '        ],',
        '        "symptom_monitoring": [',
        '            "list of symptoms patient should track and report"',
        '        ]',
        '    },',
        '    "follow_up_plan": {',
        '        "initial_follow_up": "when to follow up first (e.g., 1 week)",',
        '        "ongoing_follow_up": "subsequent schedule",',
        '        "contingencies": [',
        '            {',
        '                "condition": "situation that would require earlier follow-up",',
        '                "action": "what to do if this occurs"',
        '            }',
        '        ]',
        '    },',
        '    "patient_education": {',
        '        "key_points": ["list of 3-5 most important things patient should understand"],',
        '        "medication_instructions": ["specific instructions about how/when to take each med"],',
        '        "warning_signs": ["symptoms that should prompt immediate medical attention"]',
        '    },',
        '    "referrals": [',
        '        {',
        '            "specialty": "type of specialist",',
        '            "urgency": "when referral should happen",',
        '            "reason": "why referral is needed"',
        '        }',
        '    ]',
        "}",
        "",
        "Recommendations should be:",
        f"- Specific to {specialty} practice guidelines",
        "- Evidence-based and tailored to this patient's profile",
        "- Include exact dosages, frequencies, and durations",
        "- Provide clear monitoring parameters",
        "- Include specific follow-up timing",
        "- Highlight critical interactions and how to manage them",
        "- Return ONLY the JSON object with no additional text"
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
            'frequency': m.frequency
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
        # Include all doctor-specific outputs in structured format
        rec_data = state['recommendations'].dict()
        result.update({
            'clinical_assessment': rec_data.get('clinical_assessment', {}),
            'medication_plan': rec_data.get('medication_plan', {}),
            'monitoring_plan': rec_data.get('monitoring_plan', {}),
            'follow_up_plan': rec_data.get('follow_up_plan', {}),
            'patient_education': rec_data.get('patient_education', {}),
            'referrals': rec_data.get('referrals', [])
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

    patient_data = {
        "Blood_Pressure": patient.get('bloodPressure'),
        "Age": patient.get('anchorAge'),
        "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek'),
        "Diet":  patient.get('diet'),
        "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay'),
        "Stress_Level": patient.get('stressLevel'),
        "glucose": patient.get('glucose'),
        "BMI": patient.get('bmi'),
        "hypertension":  1 if patient.get("bloodPressure", 0) > 130 else 0,
        "is_smoking": patient.get('isSmoker'),
        "hemoglobin_a1c": patient.get('hemoglobinA1c'),
        "Diabetes_pedigree": patient.get('diabetesPedigree'),
        "CVD_Family_History": patient.get('ckdFamilyHistory'),
        "ld_value": patient.get('cholesterolLDL'),
        "admission_tsh": patient.get('admissionSOH'),
        "is_alcohol_user": patient.get('isAlcoholUser'),
        "creatine_kinase_ck": patient.get('creatineKinaseCK'),
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
