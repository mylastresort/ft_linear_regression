from LinearRegression import LinearRegression


def parse_file(file):
    try:
        line = file.readline()
        if not line:
            raise Exception("parameters are not found")
        arr = line.split()
        if len(arr) != 2:
            raise Exception("parameters are malformed")
        try:
            weight = float(arr[0])
            bias = float(arr[1])
            return weight, bias
        except Exception as exp:
            raise Exception("parameters are malformed")
    except Exception as exp:
        print(f"Fatal: error parsing file - {exp}")


def main():
    model = LinearRegression()
    x = input("Enter mileage: ")
    try:
        x = int(x)
        if x < 0:
            return print(f"{x} is not a valid mileage.")
        with open("model.lr", "r") as file:
            weight, bias = parse_file(file)
            model.set_parameters(weight, bias)
        y = int(model.predict(float(x)))
        if y < 0:
            msg = f"{x} is too big to estimate.\n"
            msg += (
                f"The model predicted {y} which is not a real price. Not enough data.\n"
            )
            msg += f"Please try with a smaller mileage.\n"
            large_price = int(model.predict(-model.b / model.w))
            msg += f"Note: the price for a max mileage is {large_price} according the data."
            return print(msg)
        print(f"Estimated value for {x} is {y}")
    except Exception as exp:
        print(f"Fatal: {exp}")


if __name__ == "__main__":
    main()
