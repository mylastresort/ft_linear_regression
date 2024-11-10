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
        with open("model.lr", "r") as file:
            weight, bias = parse_file(file)
            model.set_parameters(weight, bias)
            print(f"Model precision: mse={model.mse()}")
        y = model.predict(float(x))
        if (y < 0):
            return print(f"{x} is too big to estimate")
        print(f"Estimated value for {x} is {int(y)}")
    except:
        print("Please enter a valid input. Input must be a integer only type.")


if __name__ == "__main__":
    main()
