import math
import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE = 10000000
num_bins = 200
fig = plt.figure()
matplotlib.rcParams.update({"font.size": 7})

# Y ~ U(0,1)
ax_1 = fig.add_subplot(2, 2, 1)
ax_1.set_xlabel("U(0,1)")
lower_limit = 0
upper_limit = 1
res_1 = [random.uniform(lower_limit, upper_limit) for _ in range(0, SAMPLE_SIZE)]
ax_1.hist(res_1, num_bins)

# X_0 ~ E(1)
plt.subplot(2, 2, 2)
plt.xlabel("E(1)")
lambd = 1
res_2 = [random.expovariate(lambd) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res_2, num_bins)
plt.xlim(lower_limit, upper_limit+3)

# 由均匀分布生成指数分布：X=-log Y ~ E(1)
plt.subplot(2, 2, 3)
plt.xlabel("generated distribution")
res_3 = [-math.log(y) for y in res_1]
plt.hist(res_3, num_bins)

plt.xlim(lower_limit, upper_limit+3)
plt.show()
