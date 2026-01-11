import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier

load_dotenv()

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    Now accepts user-friendly categorical values.
    """

    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[str] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age: Optional[str] = None
        self.Vehicle_Damage: Optional[str] = None

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        # Store categorical values as-is
        self.Gender = form.get("Gender")
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = float(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel"))
        self.Vintage = int(form.get("Vintage"))
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Vehicle_Damage = form.get("Vehicle_Damage")

    def get_encoded_data(self):
        """
        Transform categorical values to encoded format for model prediction.
        Returns a dictionary with encoded values matching the training data format.
        """
        gender_encoded = 1 if self.Gender == "Male" else 0

        vehicle_age_lt_1 = 1 if self.Vehicle_Age == "< 1 Year" else 0
        vehicle_age_gt_2 = 1 if self.Vehicle_Age == "> 2 Years" else 0

        vehicle_damage_encoded = 1 if self.Vehicle_Damage == "Yes" else 0

        return {
            "Gender": gender_encoded,
            "Age": self.Age,
            "Driving_License": self.Driving_License,
            "Region_Code": self.Region_Code,
            "Previously_Insured": self.Previously_Insured,
            "Annual_Premium": self.Annual_Premium,
            "Policy_Sales_Channel": self.Policy_Sales_Channel,
            "Vintage": self.Vintage,
            "Vehicle_Age_lt_1_Year": vehicle_age_lt_1,
            "Vehicle_Age_gt_2_Years": vehicle_age_gt_2,
            "Vehicle_Damage_Yes": vehicle_damage_encoded,
        }


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """

    # Build basic MLOps status to display on UI
    def find_latest_artifact(base_dir: str):
        try:
            if not os.path.exists(base_dir):
                return None
            items = []
            for name in os.listdir(base_dir):
                full = os.path.join(base_dir, name)
                if not os.path.isdir(full):
                    continue
                try:
                    dt = datetime.strptime(name, "%m_%d_%Y_%H_%M_%S")
                    items.append((dt, full, name))
                except ValueError:
                    continue
            if not items:
                return None
            items.sort(key=lambda x: x[0], reverse=True)
            return items[0]
        except Exception:
            return None

    latest = find_latest_artifact("artifact")
    latest_dir = latest[1] if latest else None
    latest_name = latest[2] if latest else None

    def exists(rel_path: str) -> bool:
        return bool(latest_dir and os.path.exists(os.path.join(latest_dir, rel_path)))

    mlops = {
        "artifact": latest_name,
        "stages": {
            "data_ingestion": exists(os.path.join("data_ingestion", "ingested", "train.csv"))
            and exists(os.path.join("data_ingestion", "ingested", "test.csv")),
            "data_validation": exists(os.path.join("data_validation", "report.yaml")),
            "data_transformation": exists(os.path.join("data_transformation", "transformed_object", "preprocessing.pkl")),
            "model_trainer": exists(os.path.join("model_trainer", "trained_model", "model.pkl")),
        },
    }

    return templates.TemplateResponse(
        "vehicledata.html", {"request": request, "context": "Rendering", "score": None, "mlops": mlops}
    )


@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        from src.pipline.training_pipeline import TrainPipeline

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/health")
async def health():
    """Return health status including artifact availability."""
    try:
        from src.pipline.prediction_pipeline import VehicleDataClassifier

        VehicleDataClassifier()
        # Check files exist by accessing paths
        return {"status": True}
    except Exception as e:
        return {"status": False, "error": str(e)}


@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        encoded_data = form.get_encoded_data()

        vehicle_data = VehicleData(
            Gender=encoded_data["Gender"],
            Age=encoded_data["Age"],
            Driving_License=encoded_data["Driving_License"],
            Region_Code=encoded_data["Region_Code"],
            Previously_Insured=encoded_data["Previously_Insured"],
            Annual_Premium=encoded_data["Annual_Premium"],
            Policy_Sales_Channel=encoded_data["Policy_Sales_Channel"],
            Vintage=encoded_data["Vintage"],
            Vehicle_Age_lt_1_Year=encoded_data["Vehicle_Age_lt_1_Year"],
            Vehicle_Age_gt_2_Years=encoded_data["Vehicle_Age_gt_2_Years"],
            Vehicle_Damage_Yes=encoded_data["Vehicle_Damage_Yes"],
        )

        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        model_predictor = VehicleDataClassifier()

        value = model_predictor.predict(dataframe=vehicle_df)[0]
        score = None
        try:
            prob = model_predictor.predict_proba(dataframe=vehicle_df)
            score = float(prob)
        except Exception:
            score = None

        status = "Response-Yes" if int(value) == 1 else "Response-No"

        # Pass form data back to template to preserve user selections
        form_data = {
            "Gender": form.Gender,
            "Age": form.Age,
            "Driving_License": form.Driving_License,
            "Region_Code": form.Region_Code,
            "Previously_Insured": form.Previously_Insured,
            "Annual_Premium": form.Annual_Premium,
            "Policy_Sales_Channel": form.Policy_Sales_Channel,
            "Vintage": form.Vintage,
            "Vehicle_Age": form.Vehicle_Age,
            "Vehicle_Damage": form.Vehicle_Damage,
        }

        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": status, "score": score, "form_data": form_data},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
