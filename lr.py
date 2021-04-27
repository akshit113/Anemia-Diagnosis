from pandas import read_csv
from pandas_profiling import ProfileReport


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


def visualize(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file('profile.html')
    print('')


if __name__ == '__main__':
    fpath = 'dataset/data.csv'
    df = import_data(fname=fpath)
    visualize(df)
    print(list(df.columns))

    print(df.head())

    print('')
