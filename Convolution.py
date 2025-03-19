import numpy as np

def conv2d(X, F, s=1, p=0):
    """
    X: matrix input
    F: filter
    s: step jump
    p: padding
    """

    (W, H) = X.shape
    f = F.shape[0]
    # Output dimensions
    w = (W - f + 2 * p) // s + 1
    h = (H - f + 2 * p) // s + 1
    X_pad = np.pad(X, pad_width=((p,p),(p,p)), mode='constant', constant_values=0)
    # print(w,h)
    Y = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            x = i * s
            y = j * s
            Y[i][j] = np.sum(X_pad[x:(x+f),y:(y+f)]*F)
    return Y