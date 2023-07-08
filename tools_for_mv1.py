import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import t
from copy import deepcopy

def nan_count(a,ind):
    c = 0
    for i in range(len(a)):
        if np.isnan(a[i,ind]):
            c+=1
    return c

# Отрисовка графика
def plot_maker(X1, Y1, X2, Y2, title):
    plt.title(title)
    plt.scatter(X1, Y1, color='blue', label='known data')
    plt.scatter(X2, Y2, color='red', label='imputed data')
    plt.xlim([50, 200])
    plt.ylim([50, 200])
    plt.legend(loc='upper left')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

# Гауссовский шум, нужен для Stochastic regression imputation
def gaussian_noise(x, mu=0, std=None, random_state=8):
    if type(x)==np.float64 or type(x)==float:
        rs = np.random.default_rng(random_state)
        noise = rs.normal(mu, std, size=1)
        return x + noise
    if type(x)==np.ndarray:
        rs = np.random.default_rng(random_state)
        noise = rs.normal(mu, std, size=x.shape)
        return x + noise

# Выборка с возвращением
def the_sample(X,n):
    sample = list()
    inds = np.random.randint(0, len(X),n)
    for i in range(len(inds)):
        sample.append(X[inds[i]])
    sample = np.array(sample)
    return sample[~np.isnan(sample[:,1])]

#Генерация совершенно случайных данных
def MCAR(arr, size_mv, random_state=8):
    a = deepcopy(arr)
    h = [1]*size_mv+[0]*(len(arr)-size_mv)
    rs = np.random.default_rng(random_state)
    rs.shuffle(h)
    for i in range(len(a)):
        if h[i] == 1:
            a[i, 1] = np.nan
    return a

#Генерация случайных данных
def MAR(arr,threshold,order):
    a = deepcopy(arr)
    for i in range(len(a)):
        if order=='more':
            if a[i,0]>=threshold:
                a[i,1] = np.nan
        if order=='less':
            if a[i,0]<=threshold:
                a[i,1] = np.nan       
    return a

#Генерация неслучайных данных
def MNAR(arr,threshold,order):
    a = deepcopy(arr)
    for i in range(len(arr)):
        if order=='more':
            if a[i,1]>=threshold:
                a[i,1] = np.nan               
        if order=='less':
            if a[i,1]<=threshold:
                a[i,1] = np.nan
    return a

# Заполнение средним
def mean_fill(Y):
    Y_ = deepcopy(Y)
    Y_obs = np.delete(Y_, [np.any(i) for i in np.isnan(Y_)], axis=0)
    m = np.mean(Y_obs[:,1])
    for i in range(len(Y_)):
        if np.isnan(Y_[i,1]):
            Y_[i,1] = m
    return Y_ 

# Заполнение по коэффициентам линейной регрессии
def lr_fill(Y):
    Y_ = deepcopy(Y)
    Y_obs = np.delete(Y_, [np.any(i) for i in np.isnan(Y_)], axis=0)
    reg = linear_model.LinearRegression()
    reg.fit(Y_obs[:,0].reshape((-1,1)), Y_obs[:,1])
    for i in range(len(Y_)):
        if np.isnan(Y_[i,1]):
            Y_[i,1] = reg.coef_*Y_[i,0]+reg.intercept_
    return Y_

# Заполнение с использованием коэффициентов линейной регрессии с добавлением гауссовского шума
def slr_fill(Y):
    Y_ = deepcopy(Y)
    Y_obs = np.delete(Y_, [np.any(i) for i in np.isnan(Y_)], axis=0)
    reg = linear_model.LinearRegression()
    reg.fit(Y_obs[:,0].reshape((-1,1)), Y_obs[:,1])
    sigma = np.sqrt(np.var(Y_obs[:,1]-(reg.coef_*Y_obs[:,0]+reg.intercept_)))
    for i in range(len(Y_)):
        if np.isnan(Y_[i,1]):
            Y_[i,1] = gaussian_noise(reg.coef_*Y_[i,0]+reg.intercept_, mu=0, std=sigma)
    return Y_

# E-шаг
def e_step(Y,mu,sigma):
    cond_sigma = sigma[1]-(sigma[2]**2)/sigma[0]
    E_y2 = np.zeros(len(Y))
    for i in range(len(Y)):
        if np.isnan(Y[i,1])==True:
            E_y2[i] = mu[1]+sigma[2]/sigma[0]*(Y[i,0]-mu[0])
        else:
            E_y2[i] = Y[i,1].copy()
    E_y1 = Y[:,0].copy()
    E_y2_y2 = np.zeros(len(Y))
    for i in range(len(E_y2)):
        E_y2_y2[i] = E_y2[i]**2+cond_sigma*np.isnan(Y[i,1])
    E_y1_y1, E_y1_y2 = Y[:,0]**2, E_y2*E_y1
    return np.vstack((Y[:,0],E_y2)).T, sum(E_y1), sum(E_y2), sum(E_y1_y1), sum(E_y1_y2), sum(E_y2_y2)

# M-шаг
def m_step(Y, s1, s2, s11, s12, s22):
    mu1, mu2 = s1/len(Y), s2/len(Y)
    sigma1, sigma2, sigma12 = s11/len(Y)-mu1**2, s22/len(Y)-mu2**2, s12/len(Y)-mu1*mu2
    return np.array([mu1, mu2]), np.array([sigma1, sigma2, sigma12])

# Значение правдоподобия
def likelihood(Y): 
    mu1, mu2 = np.mean(Y[:,0]), np.mean(Y[:,1])
    sigma11, sigma12, sigma22 = np.var(Y[:,0]), np.mean(Y[:,0]*Y[:,1])-mu1*mu2, np.var(Y[:,1]) 
    n, r = len(Y), len(Y)
    first, third = -(n/2)*np.log(sigma11**2), -(r/2)*np.log(sigma22-sigma12*sigma12/sigma11)
    second = sum([(-1/2)*((Y[i,0]-mu1)**2)/(sigma11**2) for i in range(n)])
    fourth = sum([(-1/2)*((Y[i,0]-mu1-(sigma12/sigma11)*(Y[i,0]-mu1))**2)/(sigma22-sigma12*sigma12/sigma11) for i in range(r)])
    return first+second+third+fourth

# Расчет вариационной нижней границы
def vlb_computing(Y, sigma, K):
    r = sigma[2]/(np.sqrt(sigma[1]*sigma[0])+1e-10)
    print(f"likelihood(Y)={likelihood(Y)},r={r},s1={sigma[0]},s12={sigma[2]},s2={sigma[1]}")
    return (0.5+likelihood(Y)+np.log(np.sqrt(2*np.pi)*np.sqrt(1-r*r)*np.sqrt(sigma[1])+1e-10))*K

# EM-алгоритм
def em(Y, rtol=1e-3, max_iter=10, restarts=3, random_state=42):
    Y_obs = np.delete(Y, [np.any(i) for i in np.isnan(Y)], axis=0)
    K = len(Y_obs)
    Y_ = deepcopy(Y)
    rs = np.random.default_rng(random_state)
    k = np.array(rs.uniform(1/30,30,restarts))
    best_loss, best_mu, best_sigma, loss_prev = None, None, None, None  
    for i in range(restarts):
        loss, curr_rel_loss = None, None
        mu1, mu2 = np.mean(Y_[:,0]), np.mean(Y_obs[:,1])
        sigma1, sigma2, sigma12 = np.mean(Y_[:,0]**2)-mu1**2, np.mean(Y_obs[:,1]**2)-mu2**2, np.mean(Y_obs[:,0]*Y_obs[:,1])-mu1*mu2
        mu0, sigma0 = k[i]*np.array([mu1,mu2]), k[i]**2*np.array([sigma1,sigma2,sigma12])
        for j in range(max_iter):
            mu, sigma = mu0, sigma0
            Y_modified, s1, s2, s11, s12, s22 = e_step(Y_, mu, sigma)
            mu, sigma  = m_step(Y_, s1, s2, s11, s12, s22)
            loss = vlb_computing(Y_modified, sigma, K)
            sigma0, mu0 = sigma, mu
            if loss_prev != None:
                curr_rel_loss = np.abs(loss_prev - loss)/np.abs(loss_prev)
                loss_prev = loss
            else:
                loss_prev = loss
            print(f'Step: {j} ', f'Loss {loss:.2f}\n')
            if curr_rel_loss!=None and curr_rel_loss < rtol:
                break    
        if best_loss!=None:
            if loss>best_loss:
                best_loss, best_mu, best_sigma = loss, mu, sigma
        else:
            best_loss, best_mu, best_sigma = loss, mu, sigma
    Y_final = deepcopy(Y)
    for i in range(len(Y_final)):
        if np.isnan(Y_final[i,1]):
            Y_final[i,1] = best_mu[1]+best_sigma[2]/best_sigma[0]*(Y_final[i,0]-best_mu[0])
    return Y_final