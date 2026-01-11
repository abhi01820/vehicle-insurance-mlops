import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging



EXPECTED_COLUMNS = [
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
]




class TargetValueMapping:
    """
    Optional helper class if you want readable outputs later.
    """
    def __init__(self):
        self.yes: int = 1
        self.no: int = 0

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    """
    Wrapper class that holds:
    - preprocessing object (ColumnTransformer / Pipeline)
    - trained ML model
    """

    def __init__(
        self,
        preprocessing_object: Pipeline,
        trained_model_object: object
    ):
        """
        :param preprocessing_object: fitted preprocessing pipeline
        :param trained_model_object: fitted ML model
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        try:
            logging.info("Entered MyModel.predict method")

            if not isinstance(dataframe, pd.DataFrame):
                dataframe = pd.DataFrame(dataframe)

            dataframe = dataframe.reindex(columns=EXPECTED_COLUMNS)

            dataframe = dataframe.apply(pd.to_numeric, errors="coerce")

            dataframe.fillna(0, inplace=True)

            logging.info("Applying preprocessing pipeline")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Generating predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in MyModel.predict", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"MyModel(model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return self.__repr__()
