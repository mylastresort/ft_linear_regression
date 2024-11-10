import pandas as pd
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression
from MinMaxScaler import MinMaxScaler


def read_csv(path):
    data = pd.read_csv(path)
    return data


def create_feature(data, feature):
    scaler = MinMaxScaler()
    col = data[feature]
    feature_scaled = scaler.fit_transform(col)
    return col, feature_scaled, scaler


def plot_data_and_model(model, X, Y):
    plt.scatter(X, Y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color="red")
    plt.show()


def save_parameters(weight: float, bias: float):
    with open("model.lr", "w") as file:
        file.write(f"{weight} {bias}")


def main():
    data = read_csv("data.csv")

    X, X_scaled, scaler_x = create_feature(data, "km")
    y, y_scaled, scaler_y = create_feature(data, "price")

    model = LinearRegression(lr=0.01, n_iters=10000)
    model.fit(X_scaled, y_scaled)

    x1 = 0
    x2 = scaler_x.max
    y1 = scaler_y.inverse_transform(model.predict(scaler_x.transform(x1)))
    y2 = scaler_y.inverse_transform(model.predict(scaler_x.transform(x2)))

    model.w = (y2 - y1) / (x2 - x1)
    model.b = y2 - model.w * x2

    save_parameters(weight=model.w, bias=model.b)

    plot_data_and_model(model, X, y)


if __name__ == "__main__":
    main()
