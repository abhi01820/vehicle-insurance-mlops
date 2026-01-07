import sys
import pickle
import pandas as pd
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging


class VehicleData:
    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Vehicle_Age_lt_1_Year,
        Vehicle_Age_gt_2_Years,
        Vehicle_Damage_Yes
    ):
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> DataFrame:
        try:
            return DataFrame(self.get_vehicle_data_as_dict())
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        try:
            return {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes],
            }
        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self):
        try:
            # ✅ EXACT PATHS FROM YOUR artifact TREE
            self.model_path = (
                "artifact/01_07_2026_16_46_13/"
                "model_trainer/trained_model/model.pkl"
            )
            self.preprocessor_path = (
                "artifact/01_07_2026_16_46_13/"
                "data_transformation/transformed_object/preprocessing.pkl"
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Loading preprocessor and model")

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            # ✅ GUARANTEE dataframe for ColumnTransformer
            if not isinstance(dataframe, pd.DataFrame):
                dataframe = pd.DataFrame(dataframe)

            logging.info("Transforming input data")
            transformed_data = preprocessor.transform(dataframe)

            logging.info("Making prediction")
            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise MyException(e, sys)
