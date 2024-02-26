def assess_infertility_risk(bmi, menstrual_cycle, fsh_lh_ratio, prolactin):
    # Define thresholds for each variable, this is a dictionary of thresholds 
    bmi_thresholds = {'low': 23 , 'medium': 24.9 , "high":25 }
    menstrual_cycle_categories = {'low':  2 , 'medium': 4  , "high": 4}
    fsh_lh_ratio_categories = {'low':1, 'medium':0.7, 'high':0.4} #add proper thresholds
    prolactin_categories = {'low':25, 'medium':50, "high":51} 
    
    
    # Assign weights to variables
    weights = {'bmi': 0.2, 'menstrual_cycle': 0.2, 'fsh_lh_ratio': 0.3, 'prolactin': 0.3}#add the real weights

    # Map variable values to risk levels from each row for each of the 4 variables
    def map_to_category(value, thresholds):
        if value < thresholds['low']:
            return 'low'
        elif value <= thresholds['medium']:
            return 'medium'
        else:
            return 'high'

    # Map BMI to risk category
    bmi_category = map_to_category(bmi, bmi_thresholds)
    # Handle irregular menstrual cycle
    
    #the following code has to be changed to handle a threshhold as above . 
    menstrual_cycle_category = 'irregular' if menstrual_cycle not in menstrual_cycle_categories else menstrual_cycle
    # Map FSH/LH ratio to risk category, also all fake values have to enter proper ones 
    fsh_lh_ratio_category = 'low' if fsh_lh_ratio < 1 else ('high' if fsh_lh_ratio > 2 else 'medium')
    # Map prolactin levels to risk category, fake values substitute with proper ones 
    prolactin_category = 'normal' if prolactin < 20 else 'elevated'

    # Calculate cumulative risk score
    #the math here is correct , but it needs to be modified to refelect the chnages to be made before .

    cumulative_risk_score = (
        weights['bmi'] * (1 if bmi_category == 'low' else (0.5 if bmi_category == 'medium' else 0)) +
        weights['menstrual_cycle'] * (1 if menstrual_cycle_category == 'regular' else 0) +
        weights['fsh_lh_ratio'] * (1 if fsh_lh_ratio_category == 'high' else (0.5 if fsh_lh_ratio_category == 'medium' else 0)) +
        weights['prolactin'] * (1 if prolactin_category == 'elevated' else 0)
    )

    # low, mid, high risk should be numerical values when populated into the dataset.  eg: 2,4,6
    # Low_risk = 2

    # Interpret cumulative risk score
    if 0 <= cumulative_risk_score <= 0.3:
        return 'Low risk'
        # return Low_risk
    elif 0.31 <= cumulative_risk_score <= 0.6:
        return 'Medium risk'
    else:
        return 'High risk'

# Example usage
bmi = 22
menstrual_cycle = 'regular'
fsh_lh_ratio = 2.5
prolactin = 25

risk = assess_infertility_risk(bmi, menstrual_cycle, fsh_lh_ratio, prolactin)
print("Risk of infertility:",risk)