from LinearRegression import LinearRegression
from predict import parse_file
from train import read_csv


def main():
    with open("model.lr", "r") as file:
        model = LinearRegression()
        weight, bias = parse_file(file)
        model.set_parameters(weight, bias)
        data = read_csv("data.csv")
        mse = model.mse(data["km"], data["price"])
        print(f"Model precision: mse = {mse}")


if __name__ == "__main__":
    main()
