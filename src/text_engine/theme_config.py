"""Define the default workplace theme vocabulary used in review analysis.

``THEME_KEYWORDS`` maps seven theme identifiers to keywords and phrases for
physical demands, management, work pace, career growth, pay and benefits,
schedule flexibility, and workplace culture. The engine and exploratory
charts share this mapping, allowing vocabulary changes without editing their
analysis logic. The keyword matcher normalizes these strings with the same cleaning pipeline
as review text. THEME_DESCRIPTIONS supplies semantic model candidate labels.
"""

# Shared identifiers are used as table keys and plot categories. The dictionary
# is a manually supplied vocabulary: adding keywords expands matching coverage,
# while adding a key creates a new predefined category, not a discovered topic.
# Matching is case/inflection normalized by clean(), but remains substring-based.
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
# Zero-shot inserts each value into "This text discusses {}.". Topic alignment
# embeds the same descriptions to suggest how discovered topics map to these IDs.
# BERTopic clustering itself does not use either dictionary to discover topics.
# Keep IDs consistent with THEME_KEYWORDS when comparing lexical/semantic outputs;
# descriptions must be unique for the zero-shot description-to-ID lookup.
THEME_DESCRIPTIONS = {
    'physical_demand': 'physical strain and demanding manual work',
    'management': 'management and leadership',
    'work_pace': 'work speed and productivity targets',
    'career_growth': 'career development and promotion',
    'pay_benefits': 'pay and employee benefits',
    'schedule_flexibility': 'work schedules and working hours',
    'culture_people': 'workplace culture and relationships',
}

# Detailed theme_keywords dictionary
# (Expanded with many phrases and variants — you can tune this list)
theme_keywords = {
    "High-Velocity Physical Strain & Injury Risk": [
        "conveyor belt", "conveyor", "belt never stops", "pace", "high pace", "quota", "athletic",
        "lift", "lifting", "heavy lifting", "bending", "standing all day", "standing on concrete",
        "walking miles", "repetitive", "repetitive motion", "back pain", "shoulder pain", "knee pain",
        "wrist pain", "body breakdown", "fatigue", "exhaustion from work", "energy drink", "redbull",
        "cramps", "physically draining", "strained", "hurt on the job", "injury", "hurt", "sprain", "strain",
        "long-term body damage", "osha", "injury rate"
    ],

    "Surveillance-Driven Micromanagement": [
        "time off task", "tot", "written up", "written warning", "tracked", "monitor", "monitoring",
        "surveil", "camera", "cameras", "algorithmic", "algorithm", "metrics", "score", "rate",
        "productivity target", "upt", "u p t", "scored", "scanning", "scanned", "badge scan",
        "tracking every second", "micro-manage", "micromanage", "micromanaged", "policing", "policed",
        "policed bathroom", "bathroom break", "scrutiny", "logged"
    ],

    "Operational Inflexibility & Burnout": [
        "mandatory overtime", "met", "10 hour", "12 hour", "10-12 hours", "overtime", "forced overtime",
        "unpredictable schedule", "schedule change", "shift swap", "point system", "attendance points",
        "attendance policy", "no flexibility", "no time for family", "work-life balance", "too tired for life",
        "burnout", "exhausted after shift", "no recovery time", "cannot take leave", "pto denied", "unpaid time off",
        "upt", "unpaid"
    ],

    "Rigid Pay Structure & Caps": [
        "step plan", "step increase", "three year cap", "pay cap", "no raises", "incremental raise",
        "stuck at", "pay freeze", "wage cap", "hourly cap", "no merit", "pay plateau", "raise denied",
        "salary cap", "compensation structure", "comp plan"
    ],

    "Temporary vs. Long-Term Career": [
        "seasonal", "temporary", "temp", "contract", "one year", "two years", "short-term", "stepping stone",
        "no career", "no future", "retention", "turnover", "quit after season"
    ],

    "PA Advancement / Promotion Bottleneck": [
        "process assistant", "pa role", "promotion to pa", "promotion criteria", "hard to get promoted",
        "promotion blocked", "no clear path", "promotion unfair", "seniority only"
    ],

    "Safety & Injury Reporting": [
        "osha", "reporting", "incident report", "near miss", "ppe", "safety training", "no ppe", "unsafe",
        "hazard", "unsafe equipment", "not reported", "cover up", "safety walk", "safety protocol",
        "slip", "trip", "fall"
    ],

    "Compensation & Benefits": [
        "pay", "wage", "salary", "starting salary", "competitive pay", "benefits", "healthcare", "medical",
        "401k", "retirement", "vacation", "paid time off", "day 1 benefits", "insurance", "bonus",
        "holiday pay", "sick pay"
    ],

    "Training Gaps": [
        "training", "onboarding", "no onboarding", "no training", "we had to train each other", "no support",
        "confused", "learn on the job", "left alone", "no mentor", "cross-training", "no assigned trainer",
        "no one assigned to us", "first day invisible", "felt invisible", "no orientation"
    ],

    "Scheduling Issues": [
        "shift", "hours", "schedule", "night shift", "weekend", "shift swap", "prefered schedule denied",
        "schedule posted late", "short notice", "last minute", "unpredictable", "rota", "roster", "on-call",
        "shift cancellation"
    ],

    "Work Conditions (Physical)": [
        "cold warehouse", "hot floor", "no ventilation", "no chairs", "concrete floors", "wet floor", "dirty",
        "smelly", "lighting", "noise", "broken equipment", "standing", "no break area", "broken restroom",
        "bathroom conditions"
    ],

    "Emotional Tone & Voice": [
        "felt invisible", "i felt", "i was", "angry", "frustrated", "upset", "happy", "grateful", "appreciated",
        "proud", "demoralized", "dehumanized", "disrespected", "scared", "afraid", "unsafe", "stress",
        "stressed", "anxious", "depressed"
    ],

    "Management Support & Communication": [
        "manager", "supervisor", "no support", "ignored", "only interacts to enforce", "never heard from manager",
        "lack of leadership", "feedback never given", "no recognition", "sends emails only", "communication breakdown",
        "questions ignored", "escalation ignored"
    ],

    "Praise & Positive Signals": [
        "good pay", "solid option", "clean working environment", "day 1 benefits", "helpful teammates",
        "teamwork", "learned a lot", "fast promotion", "flexible manager", "great benefits", "positive",
        "recommend", "satisfied", "enjoyed"
    ]
}