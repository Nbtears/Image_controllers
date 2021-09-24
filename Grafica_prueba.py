from matplotlib import animation
import matplotlib.pyplot as plt
import random
from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()

def animate():
    x_vals.append(next(index))
    y_vals.append(random.randint(0,5))

    plt.cla()
    plt.plot(x_vals,y_vals)

ani = FuncAnimation(plt.gcf(),animate,interval=1000)

plt.tight_layout()
plt.show()



"""x_vals.append(angle)
    if len(angle)>5:
       x_vals.pop(0) 
    y_vals.append(len(x_vals))

    plt.cla()
    plt.plot(x_vals,y_vals)"""