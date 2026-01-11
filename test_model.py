import pickle
import pandas as pd
import os

# Load the model
model_path = os.path.join('artifact', '01_07_2026_21_36_19', 'model_trainer', 'trained_model', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("="*60)
print("MODEL INFORMATION")
print("="*60)
print(f"Model type: {type(model)}")
print(f"Has preprocessing_object: {hasattr(model, 'preprocessing_object')}")
print(f"Has trained_model_object: {hasattr(model, 'trained_model_object')}")

if hasattr(model, 'trained_model_object'):
    print(f"Base model type: {type(model.trained_model_object)}")
    
print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

# Test case 1: High probability for positive response
# Vehicle damaged, high premium, previously not insured
test_data_1 = pd.DataFrame({
    'Gender': [1],  # Male
    'Age': [40],
    'Driving_License': [1],  # Yes
    'Region_Code': [28.0],
    'Previously_Insured': [0],  # NOT previously insured (more likely to buy)
    'Annual_Premium': [55555.0],  # High premium
    'Policy_Sales_Channel': [26.0],
    'Vintage': [520],
    'Vehicle_Age_lt_1_Year': [0],
    'Vehicle_Age_gt_2_Years': [1],  # Old vehicle
    'Vehicle_Damage_Yes': [1]  # Vehicle damaged (more likely to buy)
})

pred_1 = model.predict(test_data_1)
print(f"\nTest 1 (Vehicle Damage=Yes, Previously_Insured=No):")
print(f"  Prediction: {pred_1[0]} ({'Response-Yes' if int(pred_1[0]) == 1 else 'Response-No'})")

# Test case 2: Low probability for positive response
# No vehicle damage, already insured
test_data_2 = pd.DataFrame({
    'Gender': [0],  # Female
    'Age': [25],
    'Driving_License': [1],
    'Region_Code': [28.0],
    'Previously_Insured': [1],  # Already insured (less likely to buy)
    'Annual_Premium': [25000.0],
    'Policy_Sales_Channel': [26.0],
    'Vintage': [100],
    'Vehicle_Age_lt_1_Year': [1],  # New vehicle
    'Vehicle_Age_gt_2_Years': [0],
    'Vehicle_Damage_Yes': [0]  # No vehicle damage (less likely to buy)
})

pred_2 = model.predict(test_data_2)
print(f"\nTest 2 (Vehicle Damage=No, Previously_Insured=Yes):")
print(f"  Prediction: {pred_2[0]} ({'Response-Yes' if int(pred_2[0]) == 1 else 'Response-No'})")

# Test case 3: From user's screenshot
test_data_3 = pd.DataFrame({
    'Gender': [1],
    'Age': [40],
    'Driving_License': [1],
    'Region_Code': [46.0],
    'Previously_Insured': [1],  # Yes
    'Annual_Premium': [55555.0],
    'Policy_Sales_Channel': [26.0],
    'Vintage': [520],
    'Vehicle_Age_lt_1_Year': [0],
    'Vehicle_Age_gt_2_Years': [1],
    'Vehicle_Damage_Yes': [0]  # No
})

pred_3 = model.predict(test_data_3)
print(f"\nTest 3 (From Screenshot - Vehicle Damage=No, Previously_Insured=Yes):")
print(f"  Prediction: {pred_3[0]} ({'Response-Yes' if int(pred_3[0]) == 1 else 'Response-No'})")

# Check if model supports predict_proba
if hasattr(model, 'trained_model_object'):
    base_model = model.trained_model_object
    if hasattr(base_model, 'predict_proba'):
        print("\n" + "="*60)
        print("PROBABILITY SCORES")
        print("="*60)
        preprocessor = model.preprocessing_object
        
        for i, data in enumerate([test_data_1, test_data_2, test_data_3], 1):
            transformed = preprocessor.transform(data)
            proba = base_model.predict_proba(transformed)
            print(f"\nTest {i}:")
            print(f"  Prob(Response-No): {proba[0][0]:.4f}")
            print(f"  Prob(Response-Yes): {proba[0][1]:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
