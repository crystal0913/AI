import matplotlib.pyplot as plt
import numpy as np

a = np.zeros((10, 10))
a[0][2] = 1
a[0][3] = 1
a[4][5] = 1
plt.imshow(a, interpolation='none', cmap='gray')
plt.show()
print("over")
