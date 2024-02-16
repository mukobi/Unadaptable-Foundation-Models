import matplotlib.pyplot as plt

# Assuming x and y are your arrays:
x = [
    1,
    2,
    2,
    3,
    4,
]  # Example x values, note the repeated value for a vertical alignment
y = [5, 6, 7, 8, 9]  # Corresponding y values

# Now, plot it
plt.plot(x, y)

# Adding titles and labels for clarity
plt.title("Line Plot without Confidence Interval")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()
