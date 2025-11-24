
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    """Main function."""
    print("Hello, World!")
    print(sys.executable)
    print(sys.path)

    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Create the plot
    plt.plot(x, y, label="y = 2x")

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Simple Line Plot")

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()