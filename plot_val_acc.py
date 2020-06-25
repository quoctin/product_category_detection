import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
if __name__ == "__main__":
    f = open('model/model/log.pkl', 'rb')
    data = pickle.load(f)
    n = len(data['val_err'])
    print(n)
    x = np.arange(500, n*500+1, 500)
    print(x)
    plt.plot(x, data['val_err'])
    plt.show()
    plt.xticks(x)
    f.close()