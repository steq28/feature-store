from hsml.schema import Schema
from hsml.model_schema import ModelSchema

def deploy_model(project, description, X_train, y_train, version, model_name, accuracy):
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