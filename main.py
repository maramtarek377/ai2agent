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

class MedicationRecommendation(BaseModel):
    name: str
    dosage: str
    frequency: str
    reason: str

class LabRecommendation(BaseModel):
    test_name: str
    frequency: str
    reason: str

class QuestionRecommendation(BaseModel):
    question: str
    reason: str

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[dict] = None
    exercise_plan: Optional[dict] = None
    nutrition_targets: Optional[dict] = None
    doctor_recommendations: Optional[List[str]] = None
    medication_recommendations: Optional[List[MedicationRecommendation]] = None
    lab_recommendations: Optional[List[LabRecommendation]] = None
    question_recommendations: Optional[List[QuestionRecommendation]] = None

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
    """Generate dynamic prompt for specialist recommendations with detailed medication analysis"""
    patient_profile = build_patient_profile(input_values)
    
    # Filter medicines by specialty and patient conditions
    specialty_meds = []
    for m in available_meds:
        if specialty.lower() in m.specialization.lower():
            # Additional filtering based on patient conditions
            if specialty.lower() == "cardiology":
                if input_values.get('hypertension') or parse_probability(risk_probs['Heart Disease']) > 0.2:
                    specialty_meds.append(m)
            elif specialty.lower() == "endocrinology":
                if parse_probability(risk_probs['Diabetes']) > 0.2 or input_values.get('BMI', 0) > 25:
                    specialty_meds.append(m)
    
    # Build condition-specific medication recommendations
    condition_based_recommendations = []
    if specialty.lower() == "cardiology":
        if input_values.get('hypertension'):
            condition_based_recommendations.append({
                "condition": "Hypertension",
                "meds": [m for m in specialty_meds if "hypertension" in m.specialization.lower()]
            })
        if parse_probability(risk_probs['Heart Disease']) > 0.2:
            condition_based_recommendations.append({
                "condition": "High CVD Risk",
                "meds": [m for m in specialty_meds if "cholesterol" in m.specialization.lower() or "antiplatelet" in m.specialization.lower()]
            })
    elif specialty.lower() == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            condition_based_recommendations.append({
                "condition": "Diabetes",
                "meds": [m for m in specialty_meds if "diabetes" in m.specialization.lower()]
            })
        if input_values.get('BMI', 0) > 30:
            condition_based_recommendations.append({
                "condition": "Obesity",
                "meds": [m for m in specialty_meds if "obesity" in m.specialization.lower()]
            })
    
    # Build lab recommendations based on patient status
    lab_recommendations = []
    
    if specialty.lower() == "cardiology":
        if input_values.get('hypertension'):
            lab_recommendations.append({
                "test": "Basic metabolic panel",
                "frequency": "Annually",
                "rationale": "Monitor electrolytes and kidney function in hypertensive patients"
            })
        if parse_probability(risk_probs['Heart Disease']) > 0.2:
            lab_recommendations.append({
                "test": "Lipid panel",
                "frequency": "Every 6 months",
                "rationale": "Monitor LDL in high CVD risk patient"
            })
    
    elif specialty.lower() == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            lab_recommendations.append({
                "test": "Hemoglobin A1c",
                "frequency": "Every 3 months",
                "rationale": "Monitor glycemic control in high diabetes risk"
            })
    
    # Build questions based on patient inputs
    questions_to_ask = []
    
    if specialty.lower() == "cardiology":
        if input_values.get('hypertension'):
            questions_to_ask.append({
                "question": "Do you experience any chest pain or palpitations?",
                "rationale": "Assess for cardiac symptoms in hypertensive patient"
            })
        if input_values.get('is_smoking'):
            questions_to_ask.append({
                "question": "How many cigarettes do you smoke per day?",
                "rationale": "Quantify smoking habit for risk assessment"
            })
    
    elif specialty.lower() == "endocrinology":
        if parse_probability(risk_probs['Diabetes']) > 0.3:
            questions_to_ask.append({
                "question": "Do you experience increased thirst or frequent urination?",
                "rationale": "Assess for diabetes symptoms"
            })
        if input_values.get('BMI', 0) > 30:
            questions_to_ask.append({
                "question": "What weight loss methods have you tried in the past?",
                "rationale": "Understand weight history for obesity management"
            })
    
    prompt_parts = [
        f"Generate {specialty} recommendations for this patient:",
        patient_profile,
        "",
        "Health Risks:",
        f"- Diabetes: {risk_probs['Diabetes']}",
        f"- Heart Disease: {risk_probs['Heart Disease']}",
        "",
        "Current Conditions:",
        *[f"- {c['condition']}" for c in condition_based_recommendations],
        "",
        "Return recommendations in this exact JSON format:",
        "{",
        '    "medication_recommendations": [',
        '        {',
        '            "name": "Medication name from available list",',
        '            "dosage": "Specific dosage with units (e.g., 50mg)",',
        '            "frequency": "How often to take (e.g., twice daily)",',
        '            "reason": "Brief rationale for this medication"',
        '        }',
        '    ],',
        '    "lab_recommendations": [',
        '        {',
        '            "test_name": "Lab test name",',
        '            "frequency": "How often to perform",',
        '            "reason": "Clinical rationale"',
        '        }',
        '    ],',
        '    "question_recommendations": [',
        '        {',
        '            "question": "Specific question to ask",',
        '            "reason": "Why this question is important"',
        '        }',
        '    ]',
        "}",
        "",
        "Requirements:",
        f"- Only recommend medications from this list: {[m.name for m in specialty_meds]}",
        "- Base medication recommendations on:",
        *[f"  - {c['condition']}: Consider {[m.name for m in c['meds']]}" for c in condition_based_recommendations],
        "- For each medication:",
        "  - Specify exact dosage and frequency",
        "  - Provide brief clinical rationale",
        "- For labs:",
        "  - Only recommend if clinically indicated",
        "  - Specify frequency and reason",
        "- For questions:",
        "  - Make them specific to patient's conditions",
        "  - Explain why each question is important",
        "- Return ONLY the JSON object with no additional text."
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
        
        # Process recommendations into structured format
        processed_data = {}
        if sent_for == 0:
            processed_data = {
                "patient_recommendations": data.get("patient_recommendations", []),
                "diet_plan": data.get("diet_plan", {}),
                "exercise_plan": data.get("exercise_plan", {}),
                "nutrition_targets": data.get("nutrition_targets", {})
            }
        else:
            processed_data = {
                "medication_recommendations": [
                    MedicationRecommendation(**med) for med in data.get("medication_recommendations", [])
                ],
                "lab_recommendations": [
                    LabRecommendation(**lab) for lab in data.get("lab_recommendations", [])
                ],
                "question_recommendations": [
                    QuestionRecommendation(**q) for q in data.get("question_recommendations", [])
                ]
            }
        
        return {'recommendations': Recommendations(**processed_data)}
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
        # Include all specialist recommendations in structured format
        rec_data = state['recommendations'].dict()
        result.update({
            'medication_recommendations': [
                {
                    'name': m['name'],
                    'dosage': m['dosage'],
                    'frequency': m['frequency'],
                    'reason': m['reason']
                } for m in rec_data.get('medication_recommendations', [])[:3]
            ],
            'lab_recommendations': [
                {
                    'test_name': l['test_name'],
                    'frequency': l['frequency'],
                    'reason': l['reason']
                } for l in rec_data.get('lab_recommendations', [])[:3]
            ],
            'question_recommendations': [
                {
                    'question': q['question'],
                    'reason': q['reason']
                } for q in rec_data.get('question_recommendations', [])[:3]
            ]
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
