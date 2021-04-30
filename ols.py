import numpy as np
import statsmodels.api as sm
from pandas import read_csv, cut, get_dummies, concat, DataFrame, to_numeric
# from pandas_profiling import ProfileReport
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def get_bins(df, col, newcol, intervals, labels, keep=False):
    """Creates bins of a specific column in dataframe with speciifed intervals and labels
    Args:
        df: dataframe
        col: column name to be binned
        newcol: new column to created with bins
        intervals: bin intervals
        labels: bin labels
        keep: keep original values. can be true or false

    Returns: a Dataframe with bins"""

    cur_idx = list(df.columns).index(col)
    new_idx = cur_idx + 1
    df[newcol] = cut(df[col], bins=intervals, labels=labels)
    rm_df = df.pop(newcol)
    df.insert(new_idx, newcol, rm_df)
    if not keep:
        df.drop([col], inplace=True, axis=1)
    return df


def import_data(fname):
    """Import dataset and remove row numbers column
    :return: dataframe
    """
    df = read_csv(fname)
    df = df.dropna()
    df = df.reset_index(drop=True)
    cols = [x.replace(" ", "") for x in list(df.columns)]
    df.columns = cols
    df.drop(index=0, inplace=True, axis=0)
    df = df.iloc[:364, 1:]
    df = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2]]
    return df


# def visualize(df):
#     profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
#     profile.to_file('profile.html')
#     print('')


def normalize_columns(df, colnames):
    """Performs Normalization using MinMaxScaler class in Sckikit-learn"""
    scaler = MinMaxScaler()
    for col in colnames:
        x = df.loc[:, [col]]
        x = df[[col]].values.astype(float)
        x_scaled = scaler.fit_transform(x)
        df[col] = DataFrame(x_scaled)
    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')
    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """

    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col, drop_first=True)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    missing = (df.isnull().values.any())
    while missing:
        df = df.dropna()
        print(df.isnull().sum())
        missing = (df.isnull().values.any())

    print(df.shape)
    print(list(df.columns))
    print(df.shape)
    return df


def get_ols(x_train, y_train):
    model = (sm.OLS(y_train.astype(float), x_train.astype(float))).fit()
    print(model.summary())
    print(model.params)
    return model


def make_predictions(model, x_test, y_test):
    preds = model.predict(x_test.astype(float))
    combined = concat([x_test, y_test, preds], axis=1, ignore_index=True)
    preds = preds.values.tolist()
    return combined, preds


if __name__ == '__main__':
    fpath = 'dataset/data.csv'
    df = import_data(fname=fpath)
    df['RBC'] = to_numeric(df['RBC'])
    # visualize(df)
    # df = get_bins(df, col='Age', newcol='age_type', intervals=[0, 18, 30, 60, 200],
    #               labels=['young', 'adult', 'mature', 'retired'])
    # df = get_bins(df, col='Age', newcol='age_type', intervals=[0, 18, 30, 45, 60, 200],
    #               labels=['young', 'adult', 'mature', 'more matured', 'retired'])
    # df = normalize_columns(df, colnames=['Age', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT/mm3', 'HGB'])

    X = df.iloc[:, :10]
    Y = df['RBC']
    shuffle_df = df.sample(frac=1)

    # Define a size for your train set
    train_size = int(0.8 * len(df))

    # Split your dataset
    train_set = (shuffle_df[:train_size]).reset_index(drop=True)
    test_set = (shuffle_df[train_size:]).reset_index(drop=True)
    x_train, y_train = train_set.iloc[:, :10], train_set['RBC']
    x_test, y_test = test_set.iloc[:, :10], test_set['RBC']
    ols_model = get_ols(x_train, y_train)
    df, preds = make_predictions(ols_model, x_test, y_test)

    ls = [float(val) for val in y_test]
    r2 = r2_score(ls, preds)
    mse = mean_squared_error(ls, preds)

    print(f'r2 score is {r2}')
    print(f'mse is {mse}')
    print('program execution complete')


