import matplotlib.pyplot as plt
from pandas import to_numeric
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from ols import import_data, make_predictions


def get_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f'model score is {model.score(x_train, y_train)}')
    return model


if __name__ == '__main__':
    fpath = 'dataset/data.csv'
    df = import_data(fname=fpath)
    df['RBC'] = to_numeric(df['RBC'])
    # visualize(df)
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
    regressor = get_model(x_train, y_train)
    df, preds = make_predictions(regressor, x_test, y_test)
    # df.reset_index(inplace=True)
    ls = [float(val) for val in y_test]
    r2 = r2_score(ls, preds)
    mse = mean_squared_error(ls, preds)

    plt.title('RBC Count Prediction')
    # plt.plot(y_test)

    p2 = plt.plot(df['Predicted'], linestyle='--', marker='o', label='Predicted RBC')

    from matplotlib.font_manager import FontProperties

    fontP = FontProperties()
    fontP.set_size(size='small')
    p1, = plt.plot(y_test, label='Actual RBC')
    p2, = plt.plot(ls, label='Predicted RBC')
    plt.legend(handles=[p1, p2], title='RBC', bbox_to_anchor=(0.5, 0., 0.5, 0.5), loc='best')
    plt.xlabel('Observations')
    plt.ylabel('RBC Count')
    plt.grid()
    plt.show()
    # plt.savefig('plot.png')

    print(f'r2 score is {r2}')
    print(f'mse is {mse}')
    print('program execution complete')
