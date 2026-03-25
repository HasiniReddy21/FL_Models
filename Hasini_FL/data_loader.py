import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data():

    df = pd.read_excel(
        "NFHS_5_India_Districts_Factsheet_Data.xls",
        engine="xlrd"
    )

    df = df.dropna()

    numeric_df = df.select_dtypes(include=["float64","int64"])

    scaler = MinMaxScaler()

    X = scaler.fit_transform(numeric_df)

    y = X[:,0]

    return X,y



'''import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():

    # Load Excel dataset
    df = pd.read_excel("NFHS_5_India_Districts_Factsheet_Data.xls", engine="xlrd")

    # Remove missing values
    df = df.dropna()

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['float64','int64'])

    # Normalize values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(numeric_df)

    # Use first column as target
    y = X[:,0]

    return X, y'''