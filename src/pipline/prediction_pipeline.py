import sys
import os
import pickle
import pandas as pd
from pandas import DataFrame
from datetime import datetime
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
            latest_dir = self._get_latest_complete_artifact_dir("artifact")
            if latest_dir is None:
                raise Exception("No complete artifact directory found (model+preprocessor)")

            self.latest_dir = latest_dir
            self.model_path = os.path.join(
                latest_dir, "model_trainer", "trained_model", "model.pkl"
            )
            self.preprocessor_path = os.path.join(
                latest_dir,
                "data_transformation",
                "transformed_object",
                "preprocessing.pkl",
            )
        except Exception as e:
            raise MyException(e, sys)

    def _get_latest_complete_artifact_dir(self, base_dir: str) -> str | None:
        """Return the newest artifact folder (by timestamp name) that contains both model.pkl and preprocessing.pkl."""
        try:
            if not os.path.exists(base_dir):
                return None
            candidates = []
            for name in os.listdir(base_dir):
                full = os.path.join(base_dir, name)
                if not os.path.isdir(full):
                    continue
                try:
                    dt = datetime.strptime(name, "%m_%d_%Y_%H_%M_%S")
                    candidates.append((dt, full))
                except ValueError:
                    # skip non-timestamp directories
                    continue
            if not candidates:
                return None
            # newest first
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, folder in candidates:
                model_path = os.path.join(folder, "model_trainer", "trained_model", "model.pkl")
                preproc_path = os.path.join(folder, "data_transformation", "transformed_object", "preprocessing.pkl")
                if os.path.exists(model_path) and os.path.exists(preproc_path):
                    return folder
            return None
        except Exception:
            return None

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Loading model (MyModel or raw estimator)")

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            # ✅ GUARANTEE dataframe for ColumnTransformer
            if not isinstance(dataframe, pd.DataFrame):
                dataframe = pd.DataFrame(dataframe)

            # Ensure numeric dtypes for numeric columns
            for col in [
                "Gender",
                "Age",
                "Driving_License",
                "Region_Code",
                "Previously_Insured",
                "Annual_Premium",
                "Policy_Sales_Channel",
                "Vintage",
                "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes",
            ]:
                if col in dataframe.columns:
                    dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

            # If saved model is MyModel wrapper, it will handle preprocessing
            if hasattr(model, "preprocessing_object") and hasattr(model, "trained_model_object"):
                logging.info("Detected MyModel wrapper; delegating predict")
                prediction = model.predict(dataframe)
            else:
                logging.info("Detected raw estimator; loading preprocessor and transforming")
                with open(self.preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                transformed_data = preprocessor.transform(dataframe)
                prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise MyException(e, sys)

    def predict_proba(self, dataframe: pd.DataFrame) -> float:
        """Return probability of positive class if supported."""
        try:
            if not isinstance(dataframe, pd.DataFrame):
                dataframe = pd.DataFrame(dataframe)
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            for col in [
                "Gender",
                "Age",
                "Driving_License",
                "Region_Code",
                "Previously_Insured",
                "Annual_Premium",
                "Policy_Sales_Channel",
                "Vintage",
                "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes",
            ]:
                if col in dataframe.columns:
                    dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

            if hasattr(model, "preprocessing_object") and hasattr(model, "trained_model_object"):
                # MyModel case
                preprocessor = model.preprocessing_object
                base_model = model.trained_model_object
                transformed = preprocessor.transform(dataframe)
                if hasattr(base_model, "predict_proba"):
                    proba = base_model.predict_proba(transformed)
                    return float(proba[0][1])
                elif hasattr(base_model, "decision_function"):
                    import numpy as np
                    score = base_model.decision_function(transformed)[0]
                    return float(1 / (1 + np.exp(-score)))
                else:
                    raise MyException("Model does not support probability output", sys)
            else:
                # Raw estimator + external preprocessor
                with open(self.preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                transformed = preprocessor.transform(dataframe)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(transformed)
                    return float(proba[0][1])
                elif hasattr(model, "decision_function"):
                    import numpy as np
                    score = model.decision_function(transformed)[0]
                    return float(1 / (1 + np.exp(-score)))
                else:
                    raise MyException("Model does not support probability output", sys)
        except Exception as e:
            raise MyException(e, sys)
