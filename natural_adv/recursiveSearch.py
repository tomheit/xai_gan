import sys
import numpy as np
import tensorflow as tf

class RecursiveSearch:

    def __init__(self):
        pass
    # hybrid shrinking search algorithm to generate adversaries
    # https://github.com/zhengliz/natural-adversary/blob/master/image/search.py
    # "Generating Natural Adversarial Examples"
    def search(self,gen,inv,cnn,x,y,y_t=None,z=None,n_samples=5000,step=0.01,l=0.,h=10.,stop=5,p=2,verbose=False):
        x_adv, y_adv, z_adv, d_adv = None, None, None, None
        # recursion
        counter = 1
        if(z is None):
            z = inv(x)
        while(True):
            delta_z = np.random.randn(n_samples, z.shape[1]) #n_samples latent space elements
            d = np.random.rand(n_samples) * (h - l) + l
            norm_p = np.linalg.norm(delta_z, ord = p, axis = 1)
            d_norm = np.divide(d, norm_p).reshape(-1, 1)
            delta_z = np.multiply(delta_z, d_norm)
            z_tilde = z + delta_z    # z tilde
            x_tilde = gen(z_tilde)   # x tilde
            y_tilde = cnn(x_tilde)   # y tilde
            y_tilde = np.argmax(y_tilde, axis = 1) # need the class, not confidence vector
            # n_samples y_tilde values generated, get the first fitting:
            if(y_t is None):
                indices_adv = np.where(y_tilde != y)[0]
            else:
                indices_adv = np.where(y_tilde == y_t)[0]
            if len(indices_adv) == 0:       # no candidate generated
                if h - l < step:
                    break
                else:
                    l = l + (h - l) * 0.5
                    counter = 1
            else:                           # certain candidates generated
                idx_adv = indices_adv[np.argmin(d[indices_adv])]
                if y_t is None:
                    assert (y_tilde[idx_adv] != y)
                else:
                    assert (y_tilde[idx_adv] == y_t)
                if d_adv is None or d[idx_adv] < d_adv:
                    x_adv = x_tilde[idx_adv]
                    y_adv = y_tilde[idx_adv]
                    z_adv = z_tilde[idx_adv]
                    d_adv = d[idx_adv]
                    l, h = d_adv * 0.5, d_adv
                    counter = 1
                else:
                    h = l + (h - l) * 0.5
                    counter += 1
                if counter > stop or h - l < step:
                    break
        # iteration
        if(d_adv is not None):
            h = d_adv
        l = max(0., h - step)
        counter = 1
        while(counter <= stop and h > 1e-4):
            delta_z = np.random.randn(n_samples, z.shape[1])
            d = np.random.rand(n_samples) * (h - l) + l
            norm_p = np.linalg.norm(delta_z, ord = p, axis = 1)
            d_norm = np.divide(d, norm_p).reshape(-1, 1)
            delta_z = np.multiply(delta_z, d_norm)
            z_tilde = z + delta_z
            x_tilde = gen(z_tilde)
            y_tilde = cnn(x_tilde)
            y_tilde = np.argmax(y_tilde, axis = 1)
            if y_t is None:
                indices_adv = np.where(y_tilde != y)[0]
            else:
                indices_adv = np.where(y_tilde == y_t)[0]
            if len(indices_adv) == 0:
                counter += 1
            else:
                idx_adv = indices_adv[np.argmin(d[indices_adv])]
                if y_t is None:
                    assert (y_tilde[idx_adv] != y)
                else:
                    assert (y_tilde[idx_adv] == y_t)
                if d_adv is None or d[idx_adv] < d_adv:
                    x_adv = x_tilde[idx_adv]
                    y_adv = y_tilde[idx_adv]
                    z_adv = z_tilde[idx_adv]
                    d_adv = d[idx_adv]
                h = l
                l = max(0., h - step)
                counter = 1
        # return
        adversary = {'x': x, 'y': y, 'z': z,
                     'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}
        return adversary