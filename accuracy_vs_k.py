import matplotlib.pyplot as plt

k_values = [10, 20, 30, 40, 50, 75, 100]

accuracies = [40, 44, 47.22, 50, 53, 56, 57.78]

plt.plot(k_values, accuracies, marker='o')

plt.xlabel("k Value")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs k")

plt.grid(True)

plt.show()
