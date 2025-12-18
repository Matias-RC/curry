import matplotlib.pyplot as plt
import numpy as np

x = [0, 2, 5, 7, 8]
y = [0, 2.45, 4.28, 4.19, 4]

plt.plot(x, y)

plt.xlabel("n° of extra steps")
plt.ylabel("+% level solved")

plt.show()
