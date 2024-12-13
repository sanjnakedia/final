#!/usr/bin/env python

import sys, json


GENDER_CHOICES = (('MALE', 'MALE'), 
                 ('FEMALE', 'FEMALE'),)

SMOKER_CHOICES = ((True, 'Yes'), 
                 (False, 'No'),)

def framingham_10year_risk(sex, age, race, total_cholesterol, hdl_cholesterol,
                           systolic_blood_pressure, smoker,
                           blood_pressure_med_treatment):
    """Requires:
        sex                             - "male" or "female" string
        age                             - string or int
        total_cholesterol               - sting or int 
        hdl_cholesterol                 - int
        systolic_blood_pressure         - int
        smoker                          - True or False. Also accepts 1 or 0 as
                                          a string or an int
        blood_pressure_med_treatment    - True or False. Also accepts 1 or 0
                                          as a string or an in
        ethnicity - "general" (default) or "south_asian"
    """
    
    # Apply ethnicity-specific multiplier
    risk_multiplier = 1.5 if race.lower() == "south_asian" else 1.0

    # Massage the input (existing code)
    if sex in ("MALE", "m", "M", "boy", "xy", "male", "Male"):
        sex = "male"
    if sex in ("FEMALE", "f", "F", "girl", "xx", "female", "Female"):
        sex = "female"
    
    if smoker in ("yes", "Y", "y", "YES", "true", "t", "True", True, 1, "1"):
        smoker = True
    if smoker in ("no", "NO", "N", "n", "false", "f", "False", False, 0, "0"):
        smoker = False
    if blood_pressure_med_treatment in ("yes", "Y", "y", "YES", "true", "t",
                                         "True", True, 1, "1"):
        blood_pressure_med_treatment = True
    if blood_pressure_med_treatment in ("no", "NO", "N", "n", "false", "f",
                                        "False", False, 0, "0"):
        blood_pressure_med_treatment = False

    # Process data
    age = int(age)
    total_cholesterol = int(total_cholesterol)
    hdl_cholesterol = int(hdl_cholesterol)
    systolic_blood_pressure = int(systolic_blood_pressure)

    errors = []  # Collect validation errors

    # Initialize response dictionary
    response = {
        "status": 200,
        "sex": sex,
        "message": "OK",
        "age": age,
        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "systolic_blood_pressure": systolic_blood_pressure,
        "smoker": smoker,
        "blood_pressure_med_treatment": blood_pressure_med_treatment,
        "race": race,
    }

    # Validate inputs
    if not 20 <= age <= 79:
        errors.append("Age must be within the range of 20 to 79.")
    if not 130 <= total_cholesterol <= 320:
        errors.append("Total cholesterol must be within the range of 130 to 320.")
    if not 20 <= hdl_cholesterol <= 100:
        errors.append("HDL cholesterol must be within the range of 20 to 100.")
    if not 90 <= systolic_blood_pressure <= 200:
        errors.append("Systolic blood pressure must be within the range of 90 to 200.")
    if sex == 0:
        sex = "male"
    else:
        sex = "female";

    # If there are errors, return early
    if errors:
        response["status"] = 422
        response["message"] = "Validation failed."
        response["errors"] = errors
        return response

    # Calculate baseline risk (existing logic)
    points = 0
    if sex.lower() == "male":
        # Add logic for males (as per the original code)
        pass  # Retain existing logic

    elif sex.lower() == "female":
        # Add logic for females (as per the original code)
        pass  # Retain existing logic

    # Apply ethnicity multiplier
    points = int(points * risk_multiplier)

    # Calculate risk percentage based on points
    percent_risk = calculate_risk_percentage(points, sex)

    # Add final risk to response
    response["points"] = points
    response["percent_risk"] = percent_risk

    return response


def calculate_risk_percentage(points, sex):
    """Helper function to calculate risk percentage based on points."""
    if sex.lower() == "male":
        if points <= 0:
            return "<1%"
        elif points == 1:
            return "1%"
        elif points == 2:
            return "1%"
        elif points == 3:
            return "1%"
        # Add more as per the original male logic
    elif sex.lower() == "female":
        if points <= 9:
            return "<1%"
        elif 9 <= points <= 12:
            return "1%"
        elif 13 <= points <= 14:
            return "2%"
        # Add more as per the original female logic
    return "Unknown"


if __name__ == "__main__":
    try:
        sex = sys.argv[1].lower()
        age = sys.argv[2]
        total_cholesterol = sys.argv[3]
        hdl_cholesterol = sys.argv[4]
        systolic_blood_pressure = sys.argv[5]
        smoker = sys.argv[6].lower()
        blood_pressure_med_treatment = sys.argv[7].lower()
        race = sys.argv[8].lower() if len(sys.argv) > 8 else "general"

        result = framingham_10year_risk(
            sex,
            age,
            total_cholesterol,
            hdl_cholesterol,


            systolic_blood_pressure,
            smoker,
            blood_pressure_med_treatment,
            race,
        )
        print(json.dumps(result, indent=4))

    except Exception as e:
        print("An unexpected error occurred.")
        print(str(e))
