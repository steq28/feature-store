import pandas as pd

def read_dataset(path):
    df = pd.read_csv(path)

    # Visualizzare le prime righe del dataset
    print(df.head())

    # Controllare il tipo di dati e la presenza di valori nulli
    print(df.info())

    # Descrizione statistica dei dati
    print(df.describe())

    return df

def drop_null_values(df):
    df.dropna(inplace=True)
    return df

def split_departure_time_block_column(df):
    df['DEP_START_TIME'] = df['DEP_TIME_BLK'].str.slice(0, 4)
    df['DEP_END_TIME'] = df['DEP_TIME_BLK'].str.slice(5, 9)

    df = df.drop(columns=['DEP_TIME_BLK'])
    return df