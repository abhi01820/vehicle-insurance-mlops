import pytest
import pandas as pd
from src.entity.estimator import MyModel, EXPECTED_COLUMNS

def test_expected_columns():
    assert len(EXPECTED_COLUMNS) == 11
    assert "Gender" in EXPECTED_COLUMNS
    assert "Age" in EXPECTED_COLUMNS
    assert "Vehicle_Damage_Yes" in EXPECTED_COLUMNS

def test_dataframe_structure():
    test_data = pd.DataFrame({
        "Gender": [1],
        "Age": [40],
        "Driving_License": [1],
        "Region_Code": [28.0],
        "Previously_Insured": [0],
        "Annual_Premium": [55555.0],
        "Policy_Sales_Channel": [26.0],
        "Vintage": [520],
        "Vehicle_Age_lt_1_Year": [0],
        "Vehicle_Age_gt_2_Years": [1],
        "Vehicle_Damage_Yes": [1]
    })
    
    assert list(test_data.columns) == EXPECTED_COLUMNS
    assert len(test_data) == 1
