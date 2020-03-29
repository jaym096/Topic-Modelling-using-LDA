import supporting_func as func
import numpy as np

def firstDerivative(train_x, alpha, d, w):
    phi_transpose_d = np.matmul(np.transpose(train_x), d)
    alpha_w = alpha * w
    fd = phi_transpose_d - alpha_w
    return fd
    
def Hessian(train_x, alpha, R):
    phi = train_x
    Phi_transpose_R = np.matmul(np.transpose(phi), R)
    phi_R_phi = np.matmul(Phi_transpose_R, phi)
    #neg_phi_R_phi = (-1) * phi_R_phi
    I = np.identity(len(phi_R_phi))
    alpha_I = alpha * I
    sd = - phi_R_phi - alpha_I
    return sd
       
def GLM(train_x, train_y, w, alpha, algo):
    w_old = w
    iterations = 0
    while(iterations != 100):
        a = np.matmul(train_x, w_old)
        if(algo == "pos" or algo == "log"):
            y, R = func.calculateYR(a, algo)
            d = train_y - y
        if(algo == "ord"):
            d, R = func.calculateDR(a, algo, train_y)
        g = firstDerivative(train_x, alpha, d, w_old)
        H = Hessian(train_x, alpha, R)
        H_inv = np.linalg.inv(H)
        w_new = w_old - np.matmul(H_inv, g)
        #a = np.matmul(train_x, w_new)
        constraint = np.linalg.norm(w_new - w_old,2)/np.linalg.norm(w_old,2)
        if(constraint < 0.001):
            w_old = w_new
            break
        w_old = w_new
        iterations += 1
    #print(iterations)
    return w_old, iterations