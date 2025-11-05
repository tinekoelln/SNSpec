# check_stack.py — verify numpy, matplotlib, astropy
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt

# 1) numpy: simple computation
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)

# 2) astropy: quick time conversion
t = Time.now()
print("Astropy Time (UTC):", t.iso)

# 3) matplotlib: generate a plot to disk (no GUI needed)
plt.plot(x, y)
plt.title("Sine wave")
plt.xlabel("x [rad]")
plt.ylabel("sin(x)")
plt.tight_layout()
plt.savefig("sine.png")   # writes file in current directory
print("Saved plot -> sine.png")
