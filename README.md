# Hopsworks Feature Store: Predictive Modeling of Flight Departure Delays

This project utilizes Hopsworks for feature store management, aiming to predict flight departure delays using machine learning models. Hopsworks facilitates efficient feature engineering, storage, and retrieval, ensuring consistent and scalable data management throughout the machine learning workflow.

## Project Structure

The directory structure of the project is as follows:

```
feature-store/
├── config/
│   └── constants.py
├── data/
│   ├── model/
│   │   ├── mlp_model.pkl
│   │   └── nn_model.pkl
│   └── raw_dataset/
├── functions/
│   ├── A_preprocessing.py
│   ├── B_create_feature_store.py
│   ├── C_task1_model_train.py
│   ├── D_task2.py
│   └── deploy_model.py
├── .gitignore
├── createDataset.ipynb
├── main.ipynb
└── predict_example.py
```

### Directory and File Descriptions

- **config/**
  - `constants.py`: Contains constant variables and configurations used across the project.

- **data/**
  - `model/`: Stores trained machine learning models.
    - `mlp_model.pkl`: A Multi-Layer Perceptron model.
    - `nn_model.pkl`: A Neural Network model.
  - `raw_dataset/`: Directory for storing raw datasets used in the project.

- **functions/**
  - `A_preprocessing.py`: Script for preprocessing raw data, including cleaning and feature extraction.
  - `B_create_feature_store.py`: Script for creating and managing the feature store using Hopsworks.
  - `C_task1_model_train.py`: Script for training the model for Task 1 (predicting departure delays).
  - `D_task2.py`: Script for Task 2, which may include additional analysis or resource optimization tasks.
  - `deploy_model.py`: Script for deploying the trained model.

- **Jupyter Notebooks**
  - `createDataset.ipynb`: Notebook for dataset creation and exploration.
  - `main.ipynb`: Main notebook used for the overall workflow, including data processing, feature store creation, model training, and evaluation.

- **Other Files**
  - `.gitignore`: Specifies files and directories to be ignored by Git.
  - `predict_example.py`: Example script for using the trained model to make predictions.

## Hopsworks Feature Store

The core of this project is the Hopsworks feature store, which provides a centralized platform for managing feature data. This feature store simplifies feature engineering, storage, and retrieval, ensuring consistency and efficiency across different models and data pipelines.

### Key Features of Hopsworks Used:

- **Feature Groups:** Organized and structured storage of features, making them easily accessible and manageable.
- **Online/Offline Store:** Supports real-time (online) and batch (offline) feature access for various use cases.
- **Versioning:** Tracks feature versions, ensuring data consistency, reproducibility, and easy updates.

## Usage

### Prerequisites

- Python 3.6+
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)
- Hopsworks account and API access

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/steq28/feature-store.git
   cd feature-store
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Hopsworks Configuration:**
   - Configure your Hopsworks connection settings in `config/constants.py`.

4. **Run Jupyter Notebooks:**
   - Start Jupyter Notebook and open `main.ipynb` to execute the workflow.
   - Use `createDataset.ipynb` for dataset exploration and creation.

5. **Execute Python Scripts:**
   - Data preprocessing, feature store creation, and model training can be run through the scripts in the `functions/` directory.

### Running the Model

To make predictions using the trained model, run the `predict_example.py` script with the appropriate input data.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code adheres to the project's coding standards and includes relevant documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the Hopsworks team for their feature store solution, and to the open-source community for valuable resources and tools.
