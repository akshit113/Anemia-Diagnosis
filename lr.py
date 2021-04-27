from pandas import read_csv, cut, get_dummies, concat, DataFrame

# from pandas_profiling import ProfileReport
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


if __name__ == '__main__':
    fpath = 'dataset/data.csv'
    df = import_data(fname=fpath)
    # visualize(df)
    print(list(df.columns))
    df = get_bins(df, col='Age', newcol='age_type', intervals=[0, 18, 30, 60, 200],
                  labels=['young', 'adult', 'mature', 'retired'])

    df = normalize_columns(df, colnames=['PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT/mm3', 'HGB'])
    df = one_hot_encode(df, colnames=['age_type'])
    print(df.head())

    print('')
