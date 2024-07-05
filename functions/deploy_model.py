import os
import joblib


def deploy_model(project):
    mr = project.get_model_registry()

    # rf_model = mr.sklearn.create_model(
    #     name="flight_delay_model",
    #     version=1,
    #     model=model,
    #     description="Random Forest model for predicting flight delays"
    # )
    tf_model  = mr.tensorflow.create_model(
            
            name="flight_delay_model",
            version=1,
            metrics={"accuracy": 0.81},
            description="MLPClassifier model for predicting flight delays"
    )

    tf_model.save('./data/model/modello.pkl')