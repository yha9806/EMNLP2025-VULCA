import pandas as pd
import ast
from pathlib import Path
import numpy as np # Added for np.nan comparison if needed
from tqdm import tqdm # Added for progress bar
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for plotting
from sklearn.manifold import TSNE # Added for t-SNE
from sklearn.cluster import KMeans # Added for K-Means
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import dump, load
import os # Checklist 1.1: Add import os

# Attempt UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn package not found. UMAP visualization will be skipped.")

# --- Define MICRO Level Specialized Profile Criteria (NEW - Data Driven) ---
SPECIALIZED_PROFILE_CRITERIA_MICRO = {
    "博学通论型": { # NEW - Based on K-Means Cluster 4 Analysis
        "en_name": "Comprehensive Analyst",
        "main_stance_rule": None,
        "flexible_rules": [ # Using pseudo-rule approach - requires many features >= 0.6
             # Dynamically generate this in code or list all relevant features?
             # Listing explicitly for clarity, based on ALL_POSSIBLE_FEATURE_LABELS
             {"label": "Use of Color", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Brushwork Technique", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Texture Strokes (Chunfa)", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Line Quality", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Ink Application", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Layout and Structure", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Spatial Representation", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Pictorial Structure", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Artistic Conception", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Emotional Expression", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Subject Matter", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Genre", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Symbolism", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Historical Context", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Artist Biography", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Style/School", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Technique Inheritance & Innovation", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Cross-cultural Influence", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Stylistic Analysis", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Cross-cultural Comparison", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Theoretical Construction", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Critical Review", "type": "feature", "min": 0.60, "max": 1.00},
             {"label": "Artist Case Study", "type": "feature", "min": 0.60, "max": 1.00}
             # Add other features if ALL_POSSIBLE_FEATURE_LABELS contains more
        ],
        "min_flexible_rules_to_pass": 10 # Requires at least 10 features to have score >= 0.6
    },
    "历史考据型": {
        "en_name": "Historically Focused", # Profile 1 (Adjusted based on K-Means Cluster 2 tendencies)
        "main_stance_rule": None, # MODIFIED: Removed stance requirement
        "flexible_rules": [
            {"label": "Historical Context", "type": "feature", "min": 0.50, "max": 1.00}, # MODIFIED: Lowered min score from 0.60
            {"label": "Artist Biography", "type": "feature", "min": 0.40, "max": 1.00}, # Added rule
            {"label": "Style/School", "type": "feature", "min": 0.40, "max": 1.00}, # Added rule
            {"label": "Classical Citations", "type": "quality", "min": 0.25, "max": 1.00} # Kept quality rule
        ],
        "min_flexible_rules_to_pass": 2 # Kept min required
    },
    "技艺风格型": {
        "en_name": "Technique & Style Focused", # Profile 2 (Kept as is for now)
        "main_stance_rule": {"label": "Aesthetic Appreciation", "type": "stance", "min": 0.40, "max": 1.00}, # MODIFIED: Increased min score from 0.20
        "flexible_rules": [
            {"label": "Technique Inheritance & Innovation", "type": "feature", "min": 0.30, "max": 1.00},
            {"label": "Artist Case Study", "type": "feature", "min": 0.25, "max": 1.00}, # Assuming this is a valid feature label
            {"label": "Artistic Conception", "type": "feature", "min": 0.20, "max": 1.00},
            {"label": "Emotional Expression", "type": "feature", "min": 0.15, "max": 1.00}
        ],
        "min_flexible_rules_to_pass": 2
    },
    "理论比较型": {
        "en_name": "Theory & Comparison Focused", # Profile 3 (Adjusted based on K-Means Cluster 0 tendencies)
        "main_stance_rule": None, # No single defining stance
        "flexible_rules": [
            {"label": "Stylistic Analysis", "type": "feature", "min": 0.30, "max": 1.00}, # Kept
            {"label": "Cross-cultural Comparison", "type": "feature", "min": 0.40, "max": 1.00}, # Increased min score
            {"label": "Theoretical Construction", "type": "feature", "min": 0.30, "max": 1.00}, # Increased min score
            {"label": "Critical Review", "type": "feature", "min": 0.20, "max": 1.00}, # Kept
            {"label": "Layout and Structure", "type": "feature", "min": 0.50, "max": 1.00}, # MODIFIED from Pictorial Structure
            {"label": "Spatial Representation", "type": "feature", "min": 0.50, "max": 1.00}, # Added rule
            {"label": "Symbolism", "type": "feature", "min": 0.50, "max": 1.00} # Added rule
        ],
        "min_flexible_rules_to_pass": 3 # Increased min required
    }
    # Removed old "博古通今型", "尚意唯美型", "思辨求新型"
    # "综合描述型" (General Descriptive) will be handled by GDP logic or default 'Lacks_Specific_Focus'
}

# --- Define General Descriptive Profile Criteria (GDP) ---
# Kept for now to assign texts not matching specialized profiles
GENERAL_DESCRIPTIVE_PROFILE_CRITERIA = {
    "泛化描述型": {
        "en_name": "General Descriptive Profile",
        "any_primary_stance_rules": [ # At least one of these stances must be the primary stance
            {"label": "Objective Description", "min_score": 0.15},
            {"label": "Socio-cultural Interpretation", "min_score": 0.15},
            {"label": "Aesthetic Appreciation", "min_score": 0.15},
            {"label": "Historical Research", "min_score": 0.15}
        ],
        "feature_pool_rules": {
            "pool": [ # Features to consider for GDP
                "Cross-cultural Influence", "Historical Context", "Layout and Structure", # MODIFIED from Pictorial Structure
                "Symbolism", "Emotional Expression", "Subject Matter", "Use of Color", 
                "Artist Biography", "Style/School", "Brushwork Technique", "Artistic Conception"
            ],
            "min_features_to_mention": 3, # Minimum number of features from the pool that must be mentioned
            "min_average_score_for_mentioned": 0.20 # Minimum average score for these mentioned features
        }
    }
}

# --- Begin Author Name Extraction Logic ---
AUTHOR_WORKS_MAP = {
    "聂崇正_清代宫廷绘画": {"cn_author": "聂崇正", "en_author": "Nie Chongzheng", "work": "清代宫廷绘画"},
    "高居翰（James Cahill）_明清绘画中作为思想观念的风格": {"cn_author": "高居翰", "en_author": "James Cahill", "work": "明清绘画中作为思想观念的风格"},
    "乔迅（Jonathan Hay）_清初中国的绘画与现代性": {"cn_author": "乔迅", "en_author": "Jonathan Hay", "work": "清初中国的绘画与现代性"},
    "苏利文（Sullivan）_山川悠远 中国山水画艺术": {"cn_author": "苏利文", "en_author": "Sullivan", "work": "山川悠远 中国山水画艺术"},
    "洪再辛_海外中国画研究文选（1950-1987）": {"cn_author": "洪再辛", "en_author": "Hong Zaixin", "work": "海外中国画研究文选（1950-1987）"},
    "列文森（Joseph Levenson）_从绘画所见明代与清初社会中的业余爱好理想": {"cn_author": "列文森", "en_author": "Joseph Levenson", "work": "从绘画所见明代与清初社会中的业余爱好理想"},
    "谢柏轲（Jerome Silbergeld）_西方中国绘画史研究专论": {"cn_author": "谢柏轲", "en_author": "Jerome Silbergeld", "work": "西方中国绘画史研究专论"},
    "薛永年_中国绘画断代史": {"cn_author": "薛永年", "en_author": "Xue Yongnian", "work": "中国绘画断代史"},
    "薛永年_中国美术史：清代卷": {"cn_author": "薛永年", "en_author": "Xue Yongnian", "work": "中国美术史：清代卷"},
    "贡布里希_西方人的眼光—评苏立文的《永恒的象征—中国山水画艺术": {"cn_author": "贡布里希", "en_author": "Gombrich", "work": "西方人的眼光—评苏立文的《永恒的象征—中国山水画艺术"}
}

def get_author_display_name(file_id, works_map):
    """Extracts author display name (e.g., '乔迅 (Jonathan Hay)') from file_id."""
    if not isinstance(file_id, str):
        return "Unknown Author"

    cleaned_file_id = file_id
    for suffix in [".txt", ".csv", ".json"]:
        if cleaned_file_id.endswith(suffix):
            cleaned_file_id = cleaned_file_id[:-len(suffix)]

    # Attempt to find a match by iterating through the map keys (work titles)
    # This is more robust if file_id contains the work title as a prefix or exact match
    for work_key, author_info in works_map.items():
        # Check if the cleaned_file_id starts with the work_key (e.g., work title from map)
        # This assumes file_id might be like "WorkTitle_comment1.txt"
        if cleaned_file_id.startswith(work_key):
            return f"{author_info['cn_author']} ({author_info['en_author']})"

    # Fallback if no direct match based on work_key prefix
    # Try a simpler heuristic: if the file_id starts with a known author's work key directly
    # This part might be redundant if the above loop is comprehensive enough or needs adjustment
    # For now, a simple fallback, can be refined.
    if '_' in cleaned_file_id:
        potential_work_key = cleaned_file_id.rsplit('_', 1)[0]
        if potential_work_key in works_map:
            author_info = works_map[potential_work_key]
            return f"{author_info['cn_author']} ({author_info['en_author']})"

    return "Unknown Author" # Default if no author can be reliably extracted

# --- End Author Name Extraction Logic ---

# Checklist 1.2: Define get_persona_id_from_file_id function
def get_persona_id_from_file_id(file_id):
    if pd.isna(file_id) or not isinstance(file_id, str):
        return 'gemini_unknown_persona'
    try:
        normalized_path = os.path.normpath(file_id)
        parts = normalized_path.split(os.sep)
        if len(parts) > 1:
            return parts[0]
        else:
            return 'gemini_baseline'
    except Exception:
        return 'gemini_parse_error'

# Global list of all possible feature labels for vectorization
# MUST BE ENGLISH to match keys in the features dictionary from CSV
ALL_POSSIBLE_FEATURE_LABELS = [
    "Use of Color", "Brushwork Technique", "Texture Strokes (Chunfa)", "Line Quality", "Ink Application",
    "Layout and Structure", "Spatial Representation", "Artistic Conception", "Emotional Expression",
    "Subject Matter", "Genre", "Symbolism", "Historical Context", "Artist Biography",
    "Style/School", "Technique Inheritance & Innovation", "Cross-cultural Influence",
    # Adding potentially missing labels based on new rules (if they are indeed features):
    "Stylistic Analysis", "Cross-cultural Comparison", "Theoretical Construction", "Critical Review", "Artist Case Study"
]
# Need to verify if "Classical Citations" is a feature or quality. Assuming Quality based on previous structure.
# Need to define ALL_POSSIBLE_QUALITY_LABELS if 'quality' type rules are used extensively and need separate handling/vectorization.

# --- Configuration & Setup ---
# Define a list of known features that are expected to be boolean or binary-like (0/1)
# These features will be directly used without MinMax scaling if their values are already 0 or 1.
# All other features will be MinMax scaled.
KNOWN_BINARY_FEATURES = [
    # Add feature names here if they are already binary and should not be scaled
    # e.g., 'is_comparative_analysis', 'uses_specific_terminology'
]
# --- End Configuration ---

# Global list of all possible feature labels for vectorization
# MUST BE ENGLISH to match keys in the features dictionary from CSV
ALL_FEATURE_LABELS = [
    'Comprehensive Analyst', 'General Descriptive Profile', 'Historically Focused',
    'Technique & Style Focused', 'Theory & Comparison Focused', 'Notable Omission',
    'Neutral Stance', 'Positive Stance', 'Negative Stance', 'Ambivalent Stance', 'No Clear Stance'
]
# --- End Global Feature Labels ---

# --- START: Gradient/Proportional Scoring Functions ---
def get_score_from_item(item_value):
    """Extracts numerical score if item_value is a dict like {'score': val}, else returns item_value."""
    # Ensure it handles potential None values or non-dict/non-numeric types gracefully
    if isinstance(item_value, dict) and 'score' in item_value:
        score = item_value['score']
        # Check for NaN explicitly before type check
        if pd.isna(score):
            return 0.0
        return float(score) if isinstance(score, (int, float)) else 0.0 # Return 0 if score is invalid after checking NaN
    elif isinstance(item_value, (int, float)) and pd.notna(item_value): # Use notna for non-dict values too
        return float(item_value)
    return 0.0 # Default to 0.0 if not a valid score format

def calculate_stance_contribution(row, main_stance_rule):
    """Calculates the stance contribution score (0-1) for a profile."""
    if main_stance_rule is None:
        return 1.0  # No stance requirement, fully contributes

    actual_label = row.get('stance_label_en')
    # Use get_score_from_item to safely get stance score
    actual_score = get_score_from_item(row.get('score_stance')) 

    required_label = main_stance_rule.get('label')
    min_req = main_stance_rule.get('min', 0.0)
    max_req = main_stance_rule.get('max', 1.0)

    if actual_label != required_label or actual_score < min_req:
        return 0.0

    # Handle division by zero if max_req == min_req
    denominator = max_req - min_req
    if denominator == 0:
        return 1.0 if actual_score >= min_req else 0.0

    score = (actual_score - min_req) / denominator
    return max(0.0, min(1.0, score))

def calculate_flexible_contribution(row, flexible_rules, min_flexible_rules_to_pass):
    """Calculates the flexible rules contribution score (0-1)."""
    if not flexible_rules:
        # If no rules, contribution depends if min_pass is 0
        return 1.0 if min_flexible_rules_to_pass <= 0 else 0.0 

    individual_scores = []
    num_rules_met = 0
    sum_met_scores = 0.0
    total_rules = len(flexible_rules)

    for rule in flexible_rules:
        rule_label = rule['label']
        rule_type = rule['type'] # 'feature' or 'quality'
        min_req = rule.get('min', 0.0)
        max_req = rule.get('max', 1.0)
        actual_score = 0.0
        
        source_dict = None
        if rule_type == "feature":
            source_dict = row.get('features', {})
        elif rule_type == "quality":
            source_dict = row.get('quality', {})
            
        if isinstance(source_dict, dict):
            item_value = source_dict.get(rule_label)
            actual_score = get_score_from_item(item_value) # Use helper to get score safely

        individual_score = 0.0
        is_met = False
        if actual_score >= min_req:
            is_met = True
            num_rules_met += 1
            # Handle division by zero
            denominator = max_req - min_req
            if denominator == 0:
                individual_score = 1.0
            else:
                individual_score = (actual_score - min_req) / denominator
            individual_score = max(0.0, min(1.0, individual_score))
            sum_met_scores += individual_score
        
        # Store detailed results per rule if needed for debugging later
        # individual_scores.append({"label": rule_label, "score": individual_score, "is_met": is_met})

    # Basic check
    if num_rules_met < min_flexible_rules_to_pass:
        return 0.0

    # Quantity Bonus
    quantity_bonus = 0.0
    denominator_qb = total_rules - min_flexible_rules_to_pass
    if denominator_qb <= 0: # Handles case where total_rules <= min_pass
        quantity_bonus = 1.0 # If denominator is 0 or negative, meeting min_pass means full quantity bonus
    else:
        # Ensure numerator isn't negative if somehow num_rules_met < min_pass (though checked above)
        numerator_qb = max(0, num_rules_met - min_flexible_rules_to_pass)
        quantity_bonus = numerator_qb / denominator_qb
    quantity_bonus = max(0.0, min(1.0, quantity_bonus))

    # Quality Score (Average score of rules that met the minimum requirement)
    quality_score = 0.0
    if num_rules_met > 0:
        # Re-calculate sum_met_scores to ensure it only includes scores >= min_req
        # This sum was already calculated correctly above based on `is_met` check
        quality_score = sum_met_scores / num_rules_met 
    quality_score = max(0.0, min(1.0, quality_score)) 

    # Combine quantity and quality (50/50 weighting)
    flexible_score = 0.5 * quantity_bonus + 0.5 * quality_score
    return max(0.0, min(1.0, flexible_score))

def calculate_comprehensive_score(row, profile_data):
    """Calculates the score for the Comprehensive Analyst profile."""
    features_dict = row.get('features', {})
    if not isinstance(features_dict, dict): return 0.0

    # Use rule definition for thresholds, don't hardcode
    # Assuming the *first* rule in flexible_rules defines the threshold (all should be same for this profile)
    if not profile_data.get('flexible_rules'): return 0.0 # Need rules defined
    min_required_score = profile_data['flexible_rules'][0].get('min', 0.6) 
    # min_features_needed = profile_data.get('min_flexible_rules_to_pass', 10)
    # Let's use the actual rule definition for min_pass
    min_features_needed = profile_data.get('min_flexible_rules_to_pass')
    if min_features_needed is None: return 0.0 # Need min_pass defined
    
    num_high_features = 0
    sum_high_scores = 0.0

    # Iterate through the features defined *in the rule list* for consistency
    relevant_feature_labels = [rule['label'] for rule in profile_data.get('flexible_rules', []) if rule['type'] == 'feature']

    for label in relevant_feature_labels:
        item_value = features_dict.get(label)
        actual_score = get_score_from_item(item_value)
        if actual_score >= min_required_score:
            num_high_features += 1
            # Calculate contribution relative to the threshold for quality part
            denominator_qs_indiv = 1.0 - min_required_score
            if denominator_qs_indiv > 0:
                quality_contrib = (actual_score - min_required_score) / denominator_qs_indiv
            else:
                quality_contrib = 1.0 if actual_score >= min_required_score else 0.0
            sum_high_scores += max(0.0, min(1.0, quality_contrib)) # Sum the relative scores

    if num_high_features < min_features_needed:
        return 0.0

    # Quantity Bonus
    total_relevant_features = len(relevant_feature_labels)
    denominator_qb = total_relevant_features - min_features_needed
    quantity_bonus = 0.0
    if denominator_qb <= 0:
        quantity_bonus = 1.0
    else:
        numerator_qb = max(0, num_high_features - min_features_needed)
        quantity_bonus = numerator_qb / denominator_qb
    quantity_bonus = max(0.0, min(1.0, quantity_bonus))
    
    # Quality Score (Average relative score of features meeting threshold)
    quality_score = 0.0
    if num_high_features > 0:
        quality_score = sum_high_scores / num_high_features
    quality_score = max(0.0, min(1.0, quality_score))

    # Combine (50/50 weighting)
    final_score = 0.5 * quantity_bonus + 0.5 * quality_score
    return max(0.0, min(1.0, final_score))

def calculate_gdp_score(row, profile_data):
    """Calculates the score for the General Descriptive Profile."""
    # 1. Stance Satisfaction
    stance_met = False
    min_stance_score = 0.0 # Default if not specified
    allowed_stances = []
    stance_rules = profile_data.get("any_primary_stance_rules", [])
    if stance_rules:
        # Assume first rule defines the min score if multiple exist (or use average? Using first for simplicity)
        min_stance_score = stance_rules[0].get('min_score', 0.15)
        allowed_stances = [rule['label'] for rule in stance_rules]

    actual_stance_label = row.get('stance_label_en')
    actual_stance_score = get_score_from_item(row.get('score_stance')) 
    
    if actual_stance_label in allowed_stances and actual_stance_score >= min_stance_score:
        stance_met = True
        # Calculate relative stance score contribution
        denominator_ss = 1.0 - min_stance_score
        if denominator_ss > 0:
            stance_satisfaction = max(0.0, min(1.0, (actual_stance_score - min_stance_score) / denominator_ss))
        else:
            stance_satisfaction = 1.0 if actual_stance_score >= min_stance_score else 0.0
    else:
        stance_satisfaction = 0.0 # Return 0 if stance requirement not met

    # 2. Feature Pool Satisfaction
    features_dict = row.get('features', {})
    feature_pool_rules = profile_data.get("feature_pool_rules", {})
    pool_labels = feature_pool_rules.get("pool", [])
    min_features_to_mention = feature_pool_rules.get("min_features_to_mention", 3)
    min_avg_score_required = feature_pool_rules.get("min_average_score_for_mentioned", 0.20)

    mentioned_scores_in_pool = []
    num_mentioned = 0
    if isinstance(features_dict, dict):
        for label in pool_labels:
            item_value = features_dict.get(label)
            # Consider mentioned if key exists and score is not NaN/None (get_score handles None/NaN -> 0.0)
            # We need to check if the label *exists* in the dict for the count,
            # but only average non-zero scores? Let's clarify the logic.
            # PLAN says: "calculate num_mentioned and avg_score_mentioned".
            # Let's count features present in the dict *within the pool*
            # and average their scores.
            if label in features_dict: # Check presence in the row's feature dict
                num_mentioned += 1
                actual_score = get_score_from_item(item_value) # Get score (handles bad data)
                mentioned_scores_in_pool.append(actual_score) # Include 0 scores in avg

    # Check quantity prerequisite FIRST
    if num_mentioned < min_features_to_mention:
        feature_satisfaction = 0.0 # Fails prerequisite, regardless of score
    else:
        # Calculate average score ONLY if prerequisite is met
        avg_score_mentioned = sum(mentioned_scores_in_pool) / len(mentioned_scores_in_pool) if mentioned_scores_in_pool else 0.0

        # Calculate Feature Satisfaction based on PLAN (weighted quantity/quality)
        # Quantity part - proportional to how many features are mentioned beyond minimum, up to the total pool size?
        # Let's refine: If min_features_to_mention is met, focus on quality.
        # Plan calculation:
        # quantity_part = min(1, num_mentioned / min_features_to_mention) # Always >= 1 if prerequisite met
        # quality_part = max(0, min(1, (avg_score_mentioned - min_avg_score_required) / (1.0 - min_avg_score_required))) # Handles division by zero if min_avg is 1
        
        # Revised Feature Satisfaction based on quality only if quantity met (simpler interpretation):
        feature_satisfaction = 0.0
        denominator_qs = 1.0 - min_avg_score_required
        if denominator_qs > 0:
            quality_part = (avg_score_mentioned - min_avg_score_required) / denominator_qs
        else: # min_avg_score_required is 1.0
            quality_part = 1.0 if avg_score_mentioned >= min_avg_score_required else 0.0
        feature_satisfaction = max(0.0, min(1.0, quality_part)) 

    # Combine stance and feature satisfaction (50/50 weighting as per plan)
    final_score = 0.5 * stance_satisfaction + 0.5 * feature_satisfaction
    # Consider multiplicative only if *both* must pass a threshold?
    # Sticking to additive based on plan.

    return max(0.0, min(1.0, final_score))

def calculate_profile_match_score(row, profile_data): # Removed unused all_profile_definitions arg for now
    """Calculates the overall match score (0-1) for a text chunk against a profile definition."""
    
    profile_name = profile_data.get("en_name", "Unknown")

    # Handle special profiles first based on name
    if profile_name == "Comprehensive Analyst":
        return calculate_comprehensive_score(row, profile_data) 
    if profile_name == "General Descriptive Profile":
        return calculate_gdp_score(row, profile_data)

    # Generic profile calculation for other specialized profiles
    stance_contrib = calculate_stance_contribution(row, profile_data.get('main_stance_rule'))
    flexible_contrib = calculate_flexible_contribution(row, profile_data.get('flexible_rules', []), profile_data.get('min_flexible_rules_to_pass', 0))

    # Combine scores based on plan's weighting logic
    W_STANCE = 0.3
    W_FLEXIBLE = 0.7
    final_score = 0.0
    
    if profile_data.get('main_stance_rule') is None:
        # If no stance rule, flexible score is the primary determinant
        final_score = flexible_contrib 
    else:
        # If stance rule exists, use weighted average. 
        final_score = W_STANCE * stance_contrib + W_FLEXIBLE * flexible_contrib

    return max(0.0, min(1.0, final_score))

# --- END: Gradient/Proportional Scoring Functions ---

def prepare_feature_vectors_for_visualization(df, feature_labels_list):
    """Prepares feature vectors for t-SNE/UMAP visualization."""
    feature_vectors = []
    file_ids_for_vectors = []
    valid_indices = [] # Store indices of rows with valid feature data

    for index, row in df.iterrows():
        if not isinstance(row.get('features'), dict) or not row.get('features'): # Skip if features are not a dict or empty
            # print(f"Skipping row {index} due to missing or invalid features for vectorization.") # Optional debug print
            continue
        
        current_vector = []
        has_any_valid_feature = False
        for label in feature_labels_list:
            score = 0.0 # Default score if feature is not present
            item_value = row['features'].get(label)
            if item_value is not None:
                # Use get_score_from_item to handle nested dicts like {'score': val}
                raw_score = get_score_from_item(item_value) 
                if isinstance(raw_score, (int, float)) and pd.notnull(raw_score):
                    score = raw_score
                    has_any_valid_feature = True # Mark that this row has at least one valid feature score
            current_vector.append(score)
        
        # Only include rows that had at least one of the specified features present with a valid score
        # This avoids adding all-zero vectors for texts that don't mention any of the ALL_POSSIBLE_FEATURE_LABELS
        if has_any_valid_feature:
            feature_vectors.append(current_vector)
            file_ids_for_vectors.append(row.get('file_id', f"row_{index}"))
            valid_indices.append(index) # Keep track of original df index
        # else:
            # print(f"Skipping row {index} (file_id: {row.get('file_id', 'N/A')}) as it had no valid scores for any target feature.") # Optional debug

    if not feature_vectors: # Handle case where no valid vectors could be created
        return np.array([]), [], []
        
    return np.array(feature_vectors), file_ids_for_vectors, valid_indices

def preprocess_features(df, feature_columns):
    # Implement the logic to preprocess features based on the specified columns
    # This function should return the preprocessed DataFrame
    # For example, you might want to apply a MinMaxScaler to all features
    # or handle any other preprocessing steps
    return df

def main():
    script_dir = Path(__file__).parent
    base_result_dir = script_dir / "../../result"
    # MODIFIED: Define paths for both human and MLLM data
    input_csv_path_human = base_result_dir / "human_expert_features_consolidated.csv"
    # input_csv_path_mllm = base_result_dir / "../MLLMS/result/mllms_features_consolidated.csv" # Old relative path
    # Corrected absolute-like path from workspace root for MLLM data
    workspace_root = script_dir.parents[3] # Assuming script is in v1_lang_shining_project/experiment/human_expert/src/phase4
    # This should point to the workspace root: "I:\\\\deeplearning\\\\text collection\\\\v1_lang_shining_project"
    # So, the path for gemini data is workspace_root / "experiment/MLLMS/..."
    # input_csv_path_mllm = workspace_root / "v1_lang_shining_project/experiment/MLLMS/feedbacks/gemini2.5/analysis_results/gemini2.5_features_consolidated.csv" # OLD WRONG PATH
    input_csv_path_mllm = workspace_root / "experiment/MLLMS/feedbacks/gemini2.5/analysis_results/gemini2.5_features_consolidated.csv" # CORRECTED MLLM PATH

    output_analysis_dir = base_result_dir / "analysis_results"
    output_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output path for the new CSV meant for visualization
    # output_visualization_data_path = output_analysis_dir / "data_for_composite_visualization_human_gemini.csv" # OLD Output name
    output_visualization_data_path = output_analysis_dir / "data_for_composite_visualization_human_and_gemini.csv" # CORRECTED Output name

    print(f"Loading HUMAN EXPERT data from: {input_csv_path_human}")
    if not input_csv_path_human.exists():
        print(f"Error: HUMAN EXPERT Input CSV not found at {input_csv_path_human}")
        # Decide if script should exit or try to continue with MLLM data only
        # For now, let's assume both are needed or handle this more gracefully later
        # return 
        df_human = pd.DataFrame() # Create empty if not found, concat will handle it
    else:
        df_human = pd.read_csv(input_csv_path_human)
        df_human['source_type'] = 'human_expert'
        df_human['persona_id'] = 'human_expert' # Checklist 2.1: Add persona_id for human data
        if 'stance_score' in df_human.columns: # MODIFIED: Rename stance_score
            df_human.rename(columns={'stance_score': 'score_stance'}, inplace=True)

    print(f"Loading MLLM data from: {input_csv_path_mllm}")
    try:
        df_mllm = pd.read_csv(input_csv_path_mllm)
        # print("--- Step 1: Raw MLLM Data Inspection ---") # Removed Step 1 prints
        # print("df_mllm.head():")
        # print(df_mllm.head())
        # print("\ndf_mllm.info():")
        # df_mllm.info()
        # print("\ndf_mllm.columns.tolist():")
        # print(df_mllm.columns.tolist())
        # print("--- End Step 1 ---")

        if 'stance_score' in df_mllm.columns: # MODIFIED: Rename stance_score
            df_mllm.rename(columns={'stance_score': 'score_stance'}, inplace=True)
        # Minimal MLLM data preprocessing - ensure critical columns exist
        # Standardize 'primary_profile_or_status' if MLLM uses a different name for the main profile
        if 'primary_profile_or_status' not in df_mllm.columns:
            # Try to infer or assign a default if missing. This is a placeholder.
            if 'stance_label_en' in df_mllm.columns: 
                 df_mllm['primary_profile_or_status'] = df_mllm['stance_label_en']
            else: # Added else to ensure the column is created even if stance_label_en is missing
                 print("Warning: 'primary_profile_or_status' and 'stance_label_en' not found in MLLM data. Fill with 'Unknown'.")
                 df_mllm['primary_profile_or_status'] = 'Unknown'
        # else: # This else was incorrectly placed; primary_profile_or_status is handled if it exists or not.
                 # print("Warning: 'primary_profile_or_status' not found in MLLM data. Fill with 'Unknown'.")
                 # df_mllm['primary_profile_or_status'] = 'Unknown'

        # Ensure 'file_id' and 'text_preview' exist, or create placeholders
        if 'file_id' not in df_mllm.columns:
            df_mllm['file_id'] = "mllm_" + df_mllm.index.astype(str)
        if 'text_preview' not in df_mllm.columns:
            df_mllm['text_preview'] = "" # Placeholder

        # Select and reorder columns to match human data as much as possible for concatenation
        # We need at least 'file_id', 'text_preview', 'primary_profile_or_status', and score columns.
        # The actual feature/score columns used for vectorization are defined by 'score_cols_to_plot'
        
        # Identify MLLM score columns (assuming they start with 'score_')
        mllm_score_cols = [col for col in df_mllm.columns if col.startswith('score_')]
        
        # Minimal required columns for MLLM part before concatenation
        mllm_cols_to_keep = ['file_id', 'text_preview', 'primary_profile_or_status'] + mllm_score_cols
        
        # --- FIX: Add 'features' and 'quality' to mllm_cols_to_keep if they exist in df_mllm ---
        if 'features' in df_mllm.columns:
            mllm_cols_to_keep.append('features')
        if 'quality' in df_mllm.columns:
            mllm_cols_to_keep.append('quality')
        # --- END FIX ---

        # Add other relevant columns if they exist, like 'original_filename'
        if 'original_filename' in df_mllm.columns:
            mllm_cols_to_keep.append('original_filename')
        elif 'original_relative_file_path' in df_mllm.columns: # check for alternative
            df_mllm.rename(columns={'original_relative_file_path': 'original_filename'}, inplace=True)
            mllm_cols_to_keep.append('original_filename')

        # Filter MLLM df to only these columns, handling missing ones gracefully
        df_mllm_processed = pd.DataFrame()
        for col in mllm_cols_to_keep:
            if col in df_mllm.columns:
                df_mllm_processed[col] = df_mllm[col]
            elif col == 'primary_profile_or_status': # Already handled with default
                 df_mllm_processed[col] = df_mllm[col] 
            elif col == 'features' and col not in df_mllm.columns: # Explicitly add if missing for MLLM
                print(f"Warning: Column '{col}' (features) not found in MLLM data. Initializing with empty dicts.")
                df_mllm_processed[col] = pd.Series([{} for _ in range(len(df_mllm))])
            elif col == 'quality' and col not in df_mllm.columns: # Explicitly add if missing for MLLM
                print(f"Warning: Column '{col}' (quality) not found in MLLM data. Initializing with empty dicts.")
                df_mllm_processed[col] = pd.Series([{} for _ in range(len(df_mllm))])
            else:
                print(f"Warning: Column '{col}' not found in MLLM data. It will be missing in the combined data for MLLM entries.")
                # df_mllm_processed[col] = pd.NA # Or some other placeholder like np.nan

        df_mllm = df_mllm_processed
        df_mllm['source_type'] = 'mllm' # NEW - Standardized source type for MLLM
        # Checklist 3.1: Add persona_id for MLLM data
        if 'file_id' in df_mllm.columns:
            df_mllm['persona_id'] = df_mllm['file_id'].apply(get_persona_id_from_file_id)
        else:
            print("Warning: 'file_id' column not found in MLLM data. Cannot derive 'persona_id'. Setting to 'gemini_unknown_persona'.")
            df_mllm['persona_id'] = 'gemini_unknown_persona'
        
        print(f"Human data shape: {df_human.shape}, MLLM data shape: {df_mllm.shape}")
        
        # Combine human and MLLM data
        # Before concat, ensure columns used for scores are aligned or handled appropriately if they differ
        # For now, assume 'score_cols_to_plot' will be derived from the combined df or a predefined list
        df_combined = pd.concat([df_human, df_mllm], ignore_index=True, sort=False)
        print(f"Combined data shape: {df_combined.shape}")

        # --- Step 2: Inspect features/quality in df_combined for MLLM --- 
        # if 'features' in df_combined.columns: # Removed Step 2 Prints
        #     print("\n--- Step 2: MLLM features/quality in df_combined BEFORE ast.literal_eval ---")
        #     mllm_subset_before_eval = df_combined[df_combined['source_type'] == 'mllm']
        #     print("MLLM 'features' column head before eval:")
        #     print(mllm_subset_before_eval['features'].head())
        #     print(f"MLLM 'features' column dtype before eval: {mllm_subset_before_eval['features'].dtype}")
        #     if 'quality' in mllm_subset_before_eval.columns: # quality might not be in human data
        #          print("\nMLLM 'quality' column head before eval:")
        #          print(mllm_subset_before_eval['quality'].head())
        #          print(f"MLLM 'quality' column dtype before eval: {mllm_subset_before_eval['quality'].dtype}")
        # --- End Step 2 (Before) ---

    except FileNotFoundError:
        print(f"MLLM data file not found at {input_csv_path_mllm}. Proceeding with human data only.")
        df_combined = df_human.copy() # Use only human data if MLLM is not found
        # Ensure score_stance exists if only human data is used and it had stance_score
        if 'score_stance' not in df_combined.columns and 'stance_score' in df_combined.columns:
             df_combined.rename(columns={'stance_score': 'score_stance'}, inplace=True)
    except Exception as e:
        print(f"Error loading or processing MLLM data: {e}. Proceeding with human data only.")
        df_combined = df_human.copy()
        if 'score_stance' not in df_combined.columns and 'stance_score' in df_combined.columns:
             df_combined.rename(columns={'stance_score': 'score_stance'}, inplace=True)

    # --- MODIFIED: Parse 'features' and 'quality' columns --- 
    print("Parsing 'features' and 'quality' columns from JSON strings to dictionaries...")
    if 'features' in df_combined.columns:
        df_combined['features'] = df_combined['features'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}') else (x if isinstance(x, dict) else {})
        )
    else:
        print("Warning: 'features' column not found in df_combined. Profile scoring might be affected.")
        df_combined['features'] = pd.Series([{} for _ in range(len(df_combined))]) # Add empty dict column if missing

    if 'quality' in df_combined.columns:
        df_combined['quality'] = df_combined['quality'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}') else (x if isinstance(x, dict) else {})
        )
    else:
        print("Warning: 'quality' column not found in df_combined. Profile scoring might be affected.")
        df_combined['quality'] = pd.Series([{} for _ in range(len(df_combined))]) # Add empty dict column if missing
    print("Parsing complete.")

    # --- Step 2: Inspect features/quality in df_combined for MLLM (AFTER PARSING) --- 
    # if 'features' in df_combined.columns: # Removed Step 2 Prints
    #     print("\n--- Step 2: MLLM features/quality in df_combined AFTER ast.literal_eval ---")
    #     mllm_subset_after_eval = df_combined[df_combined['source_type'] == 'mllm']
    #     print("MLLM 'features' column head after eval (should be dicts):")
    #     # Iterate to print type of first few elements to confirm they are dicts
    #     for idx, item in mllm_subset_after_eval['features'].head().items():
    #         print(f"Index {idx}, Type: {type(item)}, Content: {str(item)[:100]}...") # Print first 100 chars
    #     
    #     if 'quality' in mllm_subset_after_eval.columns:
    #         print("\nMLLM 'quality' column head after eval (should be dicts):")
    #         for idx, item in mllm_subset_after_eval['quality'].head().items():
    #             print(f"Index {idx}, Type: {type(item)}, Content: {str(item)[:100]}...")
    # print("--- End Step 2 (After) ---")
    # --- END Step 2 Inspection ---

    # --- MODIFIED: Calculate and Add Profile Match Scores ---
    print("Calculating and adding profile match scores...")
    # Specialized Profiles
    for profile_zh, profile_def in SPECIALIZED_PROFILE_CRITERIA_MICRO.items():
        profile_en_name = profile_def['en_name']
        # Sanitize column name: replace spaces with underscores, '&' with 'and', remove others
        safe_col_name_suffix = re.sub(r'[^a-zA-Z0-9_]', '', profile_en_name.replace(' ', '_').replace('&', 'and'))
        score_col_name = f"score_{safe_col_name_suffix}"
        print(f"Calculating scores for profile: {profile_en_name} -> column: {score_col_name}")
        df_combined[score_col_name] = df_combined.apply(
            lambda row: calculate_profile_match_score(row, profile_def), axis=1
        )

    # General Descriptive Profiles
    for profile_zh, profile_def in GENERAL_DESCRIPTIVE_PROFILE_CRITERIA.items():
        profile_en_name = profile_def['en_name']
        safe_col_name_suffix = re.sub(r'[^a-zA-Z0-9_]', '', profile_en_name.replace(' ', '_').replace('&', 'and'))
        score_col_name = f"score_{safe_col_name_suffix}"
        print(f"Calculating scores for profile: {profile_en_name} -> column: {score_col_name}")
        # calculate_profile_match_score internally calls calculate_gdp_score for this profile type
        df_combined[score_col_name] = df_combined.apply(
            lambda row: calculate_profile_match_score(row, profile_def), axis=1 
        )
    print("Profile score calculation complete.")

    # Checklist Item 8: Calculate and save mean profile scores by persona_id
    profile_score_columns_for_aggregation = [col for col in df_combined.columns if col.startswith('score_')]
    if 'persona_id' in df_combined.columns and profile_score_columns_for_aggregation:
        # Ensure persona_id is suitable for groupby (e.g., no NaN if it causes issues, though groupby handles it)
        # df_combined['persona_id'].fillna('Unknown_Persona_for_Aggregation', inplace=True) # Optional: handle NaN persona_ids
        
        mean_profile_scores_by_persona = df_combined.groupby('persona_id')[profile_score_columns_for_aggregation].mean()
        
        # Define output path (using output_analysis_dir already defined in main)
        output_mean_scores_path = output_analysis_dir / "profile_scores_mean_by_persona.csv"
        
        try:
            mean_profile_scores_by_persona.to_csv(output_mean_scores_path, encoding='utf-8-sig')
            print(f"Mean profile scores by persona saved to: {output_mean_scores_path}")
        except Exception as e:
            print(f"Error saving mean profile scores by persona: {e}")
    else:
        print("Warning: 'persona_id' column or profile score columns not available. Skipping saving of mean profile scores by persona.")

    # --- Feature Engineering & Vectorization ---
    # Define score columns to be used for dimensionality reduction
    # These should be common across human and MLLM data or handled if not
    # Let's try to dynamically get all 'score_*' columns from the combined dataframe
    score_cols_to_plot = [col for col in df_combined.columns if col.startswith('score_')]
    
    if not score_cols_to_plot:
        print("Error: No 'score_*' columns found in the combined data. Cannot proceed with dimensionality reduction.")
        return

    print(f"Using the following score columns for dimensionality reduction: {score_cols_to_plot}")

    # Ensure all score columns are numeric, coercing errors to NaN
    for col in score_cols_to_plot:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

    # Drop rows where *any* of the selected score columns are NaN, as they can't be used in t-SNE/UMAP
    df_combined.dropna(subset=score_cols_to_plot, inplace=True)
    
    if df_combined.empty:
        print("Error: DataFrame is empty after dropping NaNs in score columns. Cannot proceed.")
        return

    feature_vectors = df_combined[score_cols_to_plot].values

    # --- Dimensionality Reduction (t-SNE) ---
    print("Performing t-SNE reduction...")
    tsne_perplexity = min(30, feature_vectors.shape[0] - 1) if feature_vectors.shape[0] > 1 else 5
    if tsne_perplexity <=0: tsne_perplexity = 1
        
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=300, init='pca')
    try:
        tsne_results = tsne_model.fit_transform(feature_vectors)
        df_combined['tsne_x'] = tsne_results[:, 0]
        df_combined['tsne_y'] = tsne_results[:, 1]
        print("t-SNE reduction complete.")
    except Exception as e:
        print(f"Error during t-SNE: {e}. Skipping t-SNE plot generation.")

    # --- Dimensionality Reduction (UMAP) - Optional ---
    perform_umap = True # Set to False to skip UMAP
    if perform_umap:
        print("Performing UMAP reduction...")
        try:
            reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric='euclidean')
            umap_results = reducer_umap.fit_transform(feature_vectors)
            df_combined['umap_x'] = umap_results[:, 0]
            df_combined['umap_y'] = umap_results[:, 1]
            print("UMAP reduction complete.")
        except Exception as e:
            print(f"Error during UMAP reduction: {e}. Skipping UMAP.")
            df_combined['umap_x'] = pd.NA
            df_combined['umap_y'] = pd.NA
    else:
        df_combined['umap_x'] = pd.NA
        df_combined['umap_y'] = pd.NA

    # Checklist Item 10: Calculate and save centroids in profile score t-SNE/UMAP space by persona_id
    profile_tsne_centroids_list = []
    if 'persona_id' in df_combined.columns and 'tsne_x' in df_combined.columns and 'tsne_y' in df_combined.columns:
        # Drop rows where t-SNE results might be NaN to avoid issues with mean calculation
        # This should ideally be handled by the dropna(subset=score_cols_to_plot) earlier if tsne_x/y depend on scores
        # But as an extra precaution for direct t-SNE NaNs:
        df_for_tsne_centroids = df_combined.dropna(subset=['tsne_x', 'tsne_y', 'persona_id'])
        if not df_for_tsne_centroids.empty:
            tsne_centroids_profile = df_for_tsne_centroids.groupby('persona_id')[['tsne_x', 'tsne_y']].mean().reset_index()
            tsne_centroids_profile.rename(columns={'tsne_x': 'mean_tsne_x_profile_score_space',
                                                 'tsne_y': 'mean_tsne_y_profile_score_space'}, inplace=True)
            profile_tsne_centroids_list.append(tsne_centroids_profile)
            print("Calculated t-SNE centroids in profile score space by persona.")
        else:
            print("DataFrame empty after dropping NaNs for t-SNE centroid calculation based on profile scores.")
    else:
        print("Warning: 'persona_id' or t-SNE coordinate columns not available for profile score space. Skipping t-SNE centroid calculation.")

    profile_umap_centroids_list = []
    if 'persona_id' in df_combined.columns and 'umap_x' in df_combined.columns and 'umap_y' in df_combined.columns:
        df_for_umap_centroids = df_combined.dropna(subset=['umap_x', 'umap_y', 'persona_id'])
        if not df_for_umap_centroids.empty:
            umap_centroids_profile = df_for_umap_centroids.groupby('persona_id')[['umap_x', 'umap_y']].mean().reset_index()
            umap_centroids_profile.rename(columns={'umap_x': 'mean_umap_x_profile_score_space',
                                                 'umap_y': 'mean_umap_y_profile_score_space'}, inplace=True)
            profile_umap_centroids_list.append(umap_centroids_profile)
            print("Calculated UMAP centroids in profile score space by persona.")
        else:
            print("DataFrame empty after dropping NaNs for UMAP centroid calculation based on profile scores.")
    else:
        print("Warning: 'persona_id' or UMAP coordinate columns not available for profile score space. Skipping UMAP centroid calculation.")

    if profile_tsne_centroids_list and profile_umap_centroids_list:
        all_profile_score_centroids = pd.merge(profile_tsne_centroids_list[0], profile_umap_centroids_list[0], on='persona_id', how='outer')
    elif profile_tsne_centroids_list:
        all_profile_score_centroids = profile_tsne_centroids_list[0]
    elif profile_umap_centroids_list:
        all_profile_score_centroids = profile_umap_centroids_list[0]
    else:
        all_profile_score_centroids = pd.DataFrame() # Empty if neither was computed

    if not all_profile_score_centroids.empty:
        output_profile_centroids_path = output_analysis_dir / "profile_score_space_centroids_by_persona.csv"
        try:
            all_profile_score_centroids.to_csv(output_profile_centroids_path, index=False, encoding='utf-8-sig')
            print(f"Profile score space centroids by persona saved to: {output_profile_centroids_path}")
        except Exception as e:
            print(f"Error saving profile score space centroids by persona: {e}")
    else:
        print("No profile score space centroids calculated or saved.")

    # --- START: Calculate Profile Proportions ---
    print("Calculating profile proportions...")
    profile_score_columns_for_proportion = [
        'score_Comprehensive_Analyst',
        'score_Historically_Focused',
        'score_Technique_and_Style_Focused',
        'score_Theory_and_Comparison_Focused',
        'score_General_Descriptive_Profile'
    ]

    # Ensure all defined score columns for proportion exist, create them with 0 if not.
    # This is important if some profiles were not scored for all rows earlier.
    for col in profile_score_columns_for_proportion:
        if col not in df_combined.columns:
            print(f"Warning: Score column '{col}' for proportion calculation not found. Initializing with 0.")
            df_combined[col] = 0.0
        # Ensure they are numeric, coercing errors and filling NA with 0
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(0.0)


    def calculate_and_assign_proportions_V2(row, score_cols):
        profile_scores_values = row[score_cols]
        total_score = profile_scores_values.sum()
        
        proportions_data = {}
        primary_profile_label = "Lacks_Specific_Focus"

        if total_score > 0:
            for col_name in score_cols:
                # Extract the profile name part, e.g., "Comprehensive_Analyst" from "score_Comprehensive_Analyst"
                profile_name_key = col_name.split("score_", 1)[1] if "score_" in col_name else col_name
                proportions_data[f'proportion_{profile_name_key}'] = profile_scores_values[col_name] / total_score
            
            # Determine primary profile based on the original score columns (before proportion)
            # This aligns with how primary_profile_or_status was likely determined.
            # Or, if we want primary based on highest *proportion*, we'd use the just-calculated proportions.
            # For now, let's use the max of the raw scores being considered for proportion.
            # idxmax() gives the index (column name) of the max value.
            primary_profile_raw_name = profile_scores_values.idxmax()
            primary_profile_label = primary_profile_raw_name.split("score_", 1)[1] if "score_" in primary_profile_raw_name else primary_profile_raw_name
        else:
            for col_name in score_cols:
                profile_name_key = col_name.split("score_", 1)[1] if "score_" in col_name else col_name
                proportions_data[f'proportion_{profile_name_key}'] = 0.0
        
        proportions_data['primary_profile_by_proportion'] = primary_profile_label
        return pd.Series(proportions_data)

    proportion_results_df = df_combined.apply(
        lambda row: calculate_and_assign_proportions_V2(row, profile_score_columns_for_proportion),
        axis=1
    )
    
    df_combined = pd.concat([df_combined, proportion_results_df], axis=1)
    print("Profile proportion calculation complete.")
    # --- END: Calculate Profile Proportions ---

    # Checklist Item 9: Calculate and save primary_profile_by_proportion distribution by persona_id
    if 'persona_id' in df_combined.columns and 'primary_profile_by_proportion' in df_combined.columns:
        profile_distribution_by_persona = df_combined.groupby('persona_id')['primary_profile_by_proportion'].value_counts(normalize=True).mul(100).round(2)
        # Convert to DataFrame and unstack for better CSV format
        profile_distribution_df = profile_distribution_by_persona.unstack(fill_value=0)
        
        # Define output path
        output_profile_distribution_path = output_analysis_dir / "primary_profile_distribution_by_persona.csv"
        
        try:
            profile_distribution_df.to_csv(output_profile_distribution_path, encoding='utf-8-sig')
            print(f"Primary profile distribution by persona saved to: {output_profile_distribution_path}")
        except Exception as e:
            print(f"Error saving primary profile distribution by persona: {e}")
    else:
        print("Warning: 'persona_id' or 'primary_profile_by_proportion' column not available. Skipping saving of profile distribution.")

    # --- SAVE DATA FOR EXTERNAL VISUALIZATION ---
    print(f"Saving data for external visualization to: {output_visualization_data_path}")
    try:
        # Define columns to save. Ensure all necessary columns are present.
        # Minimal columns: id, profile, source_type, tsne_x, tsne_y
        # Recommended: file_id, text_preview, primary_profile_or_status, source_type, 
        #              tsne_x, tsne_y, umap_x, umap_y, original_filename, and all score_* cols.

        cols_for_visualization = ['file_id', 'text_preview', 'primary_profile_or_status', 'source_type', 'persona_id',
                                  'tsne_x', 'tsne_y', 'umap_x', 'umap_y']
        
        # Add original_filename if it exists
        if 'original_filename' in df_combined.columns:
            cols_for_visualization.append('original_filename')
        else:
            df_combined['original_filename'] = pd.NA # Add placeholder
            cols_for_visualization.append('original_filename')


        # Add all score columns used for reduction
        cols_for_visualization.extend(score_cols_to_plot)
        
        # --- START: Add new proportion columns to save list ---
        new_proportion_col_names = [f'proportion_{col.split("score_", 1)[1] if "score_" in col else col}' for col in profile_score_columns_for_proportion]
        cols_for_visualization.extend(new_proportion_col_names)
        cols_for_visualization.append('primary_profile_by_proportion')
        # --- END: Add new proportion columns to save list ---

        # Ensure all selected columns actually exist in df_combined before trying to select them
        existing_cols_for_visualization = [col for col in cols_for_visualization if col in df_combined.columns]
        
        # Add any missing columns as NA to ensure the output CSV has a consistent structure if some were conditionally added
        for col in cols_for_visualization:
            if col not in df_combined.columns:
                 df_combined[col] = pd.NA # Add as NA if missing (e.g. umap_x if UMAP failed)
                 if col not in existing_cols_for_visualization: # also add to the list if it was entirely missing
                     existing_cols_for_visualization.append(col)


        df_to_save_for_visualization = df_combined[existing_cols_for_visualization]
        
        df_to_save_for_visualization.to_csv(output_visualization_data_path, index=False, encoding='utf-8-sig')
        print("Successfully saved data for visualization.")
    except Exception as e:
        print(f"Error saving data for visualization: {e}")
        print(f"Columns available at time of error: {df_combined.columns.tolist()}")
        print(f"Columns attempted to save: {existing_cols_for_visualization}")


    # --- OLD CSV Output (dimensionality_reduction_with_profile_scores.csv) ---
    # This part can be kept if other scripts depend on this specific CSV format,
    # or removed/modified if data_for_composite_visualization.csv supersedes it.
    # For now, let's assume it's still needed for other dependent scripts.
    # output_csv_path = output_analysis_dir / "dimensionality_reduction_with_profile_scores_human_gemini.csv" # OLD Output name
    output_csv_path = output_analysis_dir / "dimensionality_reduction_with_profile_scores_human_and_gemini.csv" # CORRECTED Output name
    print(f"Saving original dimensionality reduction results (with all original columns) to: {output_csv_path}")
    try:
        df_combined.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Dimensionality reduction results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving dimensionality reduction CSV: {e}")

    print("Script execution finished.")

if __name__ == '__main__':
    main() 