import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    seen = set()
    Kn_history = []
    Kn = 0
    with open('sorted-CollegeMsg.txt') as f:
        for i, line in enumerate(f.readlines()):
            for n1 in [int(x) for x in line.split()[0:2]]:
                if n1 not in seen:
                    seen.add(n1)
                    Kn += 1
                Kn_history.append(Kn)
    
    n = len(Kn_history)
    Kn_history = np.asarray(Kn_history)

    x = np.linspace(1, n, len(Kn_history))

    model = LinearRegression()
    
    model.fit(np.log(x[:, np.newaxis]), np.log(Kn_history))
    alpha = model.coef_
    print(-1 + 1./alpha)
    
    plt.figure(figsize=(12, 18))
    plt.plot(x, Kn_history, color='#009BFA', linewidth=5)
    plt.tick_params(labelsize=30)
    plt.xlabel("#edges", fontsize=60)
    plt.ylabel("#vertices", fontsize=60)
    #plt.ytiks(plt.gca().get_yticks()[::2])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(20)
    plt.gca().yaxis.get_offset_text().set_fontsize(20)
    plt.grid()
    plt.show()    



