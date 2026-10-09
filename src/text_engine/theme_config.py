"""Define the default workplace theme vocabulary used in review analysis.

``THEME_KEYWORDS`` maps seven theme identifiers to keywords and phrases for
physical demands, management, work pace, career growth, pay and benefits,
schedule flexibility, and workplace culture. The engine and exploratory
charts share this mapping, allowing vocabulary changes without editing their
analysis logic. The matcher checks these strings directly against cleaned,
stemmed text, so entries must be compatible with that representation to match.
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
