from pandas import read_csv, cut


# from pandas_profiling import ProfileReport


def get_bins(df, col, newcol, intervals, labels):
    """Creates bins of a specific column in dataframe with speciifed intervals and labels
    Args:
        df: dataframe
        col: column name to be binned
        newcol: new column to created with bins
        intervals: bin intervals
        labels: bin labels

    Returns: a Dataframe with bins"""

    cur_idx = list(df.columns).index(col)
    new_idx = cur_idx + 1
    df[newcol] = cut(df[col], bins=intervals, labels=labels)
    rm_df = df.pop(newcol)
    df.insert(new_idx, newcol, rm_df)
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


if __name__ == '__main__':
    fpath = 'dataset/data.csv'
    df = import_data(fname=fpath)
    # visualize(df)
    print(list(df.columns))
    df = get_bins(df, col='Age', newcol='age_type', intervals=[0, 18, 30, 60, 200],
                  labels=['young', 'adult', 'mature', 'retired'])

    print(df.head())

    print('')
