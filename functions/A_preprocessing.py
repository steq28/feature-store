from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

def read_dataset(path):
    df = pd.read_csv(path)

    print(df.info())

    return df

def remove_null_values(df):
    print("Missing data:\n", df.isna().sum())
    df.dropna(inplace=True)

def split_departure_time_block_column(df):
    df['DEP_START_TIME'] = df['DEP_TIME_BLK'].str.slice(0, 4)
    df['DEP_END_TIME'] = df['DEP_TIME_BLK'].str.slice(5, 9)

    df = df.drop(columns=['DEP_TIME_BLK'])
    return df

def create_part_of_day_column(df):
    time_blocks_order = [
        'Early Morning & Late Night',  # 0001-0559
        'Morning',                     # 0600-1159
        'Afternoon',                   # 1200-1659
        'Evening',                     # 1700-1959
        'Night'                        # 2000-2359
    ]

    df['PART_OF_DAY'] = pd.cut(
        df['DEP_TIME_BLK'].map(lambda x: int(x.split('-')[0])),
        bins=[0, 600, 1200, 1700, 2000, 2400],
        labels=time_blocks_order,
        right=False
    )

    df.drop(columns=['DEP_TIME_BLK'], inplace=True)


def show_heatmap_part_of_day(df):
    heatmap_data = pd.pivot_table(
        df,
        values='DEP_DEL15',
        index='PART_OF_DAY',
        columns='DAY_OF_WEEK',
        aggfunc='mean'
    )

    # Define the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom', ['limegreen', 'yellow', 'red'])

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, fmt='.2f', linewidths=.5)

    # Adding titles and labels
    plt.title('Heatmap of Avg. Delay Rate by Time Part of Day and Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Time Part', fontsize=14)
    plt.yticks(rotation=25)
    plt.show()

def delayed_flights(df):
    plt.figure(figsize=(5, 5))
    plt.pie(df['DEP_DEL15'].value_counts(), labels=['On Time', 'Delayed'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])

    plt.title('Delayed vs On Time Flights', fontsize=16)
    plt.axis('equal')

    plt.show()

# def clean_labels_encoder(df):
#     label_encoder = LabelEncoder()
#     colsToEncode = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'PART_OF_DAY']
#     mappingDict = {}
#     for col in colsToEncode:
#         df[f'{col}'] = label_encoder.fit_transform(df[f'{col}'])
#         mappingDict[f'{col}'] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

#     # create dict with { label: value } (funziona solo quando viene eseguito dall'inizio se no diventa { value: value })
#     return df, mappingDict

def clean_labels_encoder(df):
    list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', "PART_OF_DAY"]

    le = LabelEncoder()
    for label in list_of_labels:
        df[label] = le.fit_transform(df[label])
    return df

def scale_data(data):
    # Scale the data using a MinMaxScaler
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def prepare_data_for_ML_model(df, is_NN=False, predCol="DEP_DEL15"):
    
    df = clean_labels_encoder(df)

    X = df.drop(columns=[predCol])
    y = df[predCol]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = scale_data(X_train)
    X_test = scale_data(X_test)

    if(is_NN):
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    else:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

    return X_train, X_test, y_train, y_test