"""Define the default workplace theme vocabulary used in review analysis.

``THEME_KEYWORDS`` maps seven theme identifiers to keywords and phrases for
physical demands, management, work pace, career growth, pay and benefits,
schedule flexibility, and workplace culture. The engine and exploratory
charts share this mapping, allowing vocabulary changes without editing their
analysis logic. The keyword matcher normalizes these strings with the same cleaning pipeline
as review text. THEME_DESCRIPTIONS supplies semantic model candidate labels.
"""

THEME_KEYWORDS = {
    "physical_demand": [
        "lifting", "heavy", "body", "pain", "injury", "exhaust",
        "draining", "labor intensive", "physically"
    ],
    "management": [
        "manager", "management", "supervisor", "micromanage",
        "leadership", "boss"
    ],
    "work_pace": [
        "fast paced", "rate", "target", "quota", "points",
        "warning", "time constraint"
    ],
    "career_growth": [
        "promotion", "advancement", "growth", "career",
        "opportunity", "limited opportunities"
    ],
    "pay_benefits": [
        "pay", "wage", "salary", "benefit", "401k",
        "insurance", "pto", "holiday pay"
    ],
    "schedule_flexibility": [
        "schedule", "shift", "overtime", "hours",
        "time off", "flexible", "leave"
    ],
    "culture_people": [
        "people", "team", "coworker", "toxic", "respect",
        "environment"
    ],
}


# Semantic descriptions retain full language rather than stemmed keywords.
THEME_DESCRIPTIONS = {
    'physical_demand': 'physical strain and demanding manual work',
    'management': 'management and leadership',
    'work_pace': 'work speed and productivity targets',
    'career_growth': 'career development and promotion',
    'pay_benefits': 'pay and employee benefits',
    'schedule_flexibility': 'work schedules and working hours',
    'culture_people': 'workplace culture and relationships',
}
