from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import os

def save_model_on_hopsworks(project, description, X_train, y_train, version, model_name, accuracy):
    mr = project.get_model_registry()

    # Create input schema using X_train
    input_schema = Schema(X_train)

    # Create output schema using y_train
    output_schema = Schema(y_train)

    # Create a ModelSchema object specifying the input and output schemas
    model_schema = ModelSchema(
        input_schema=input_schema, 
        output_schema=output_schema,
    )

    # Convert the model schema to a dictionary
    model_schema.to_dict()
    

    tf_model  = mr.python.create_model(
        name=f"flight_delay_{model_name}",
        version = version,
        metrics={"accuracy": accuracy},
        description=description,
        model_schema=model_schema
    )

    tf_model.save(f'./data/model/{model_name}.pkl')

def deploy_model(project, model_name, model):
    dataset_api = project.get_dataset_api()

    # Specify the local file path of the Python script to be uploaded
    local_script_path = "predict_example.py"

    # Upload the Python script to the "Models", and overwrite if it already exists
    uploaded_file_path = dataset_api.upload(local_script_path, "Models", overwrite=True)

    # Create the full path to the uploaded script for future reference
    predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

    deployment = model.deploy(
        name=f"flightdelay{model_name}deployment",  # Specify a name for the deployment
        script_file=predictor_script_path,  # Provide the path to the Python script for prediction
    )

    # Print the name of the deployment
    print("Deployment: " + deployment.name)

    # Display information about the deployment
    deployment.describe()