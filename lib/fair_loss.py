import torch
import math
import numpy as np
import statistics
from scipy import optimize
import time

def no_fair(h, sens_attri, alpha, l):
    return torch.FloatTensor([0]).to(str(h.device)), l

def pdist_squared(sample_1, sample_2):
    """ squared pair-wise distance

    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
    Returns:
        a matrix of pair-wise distance (torch tensor, [n_1, n_2])
    """

    n_1, n_2 = sample_1.size(0), sample_2.size(0)

    sample_1 = sample_1.unsqueeze(1).repeat(1,n_2,1)
    sample_2 = sample_2.unsqueeze(0).repeat(n_1,1,1)

    distances_squared = torch.pow(sample_1 - sample_2, 2).sum(dim=-1)

    return distances_squared

def pdist_squared_parallel(sample_1, sample_2):
    """ squared pair-wise distance

    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
    Returns:
        a matrix of pair-wise distance (torch tensor, [n_1, n_2])
    """

    n_1, n_2 = sample_1.size(0), sample_2.size(0)

    start = time.time()
    torch.cdist(sample_1.unsqueeze(0), sample_2.unsqueeze(1))
    end = time.time()

    sample_1 = sample_1.unsqueeze(1).repeat(1,n_2,1)
    sample_2 = sample_2.unsqueeze(0).repeat(n_1,1,1)

    distances_squared = torch.pow(sample_1 - sample_2, 2).sum(dim=-1)

    return distances_squared

def gram_with_RBF_kernel(sample_1, sample_2, l):
    """ calculate the gram matrix for RBF kernel between two sets of samples

    k^{rbf}_{\sigma} (x, y) = 
        \exp(-\frac{1}{2*l**2} \|x, y\|^2)

    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
        sigma: the parameter in RBF kernel
    
    Returns:
        the gram matrix (torch tensor, [n_1, n_2])
    """

    distances_squared = pdist_squared(sample_1, sample_2)
    out = torch.exp(distances_squared * -1/(2*l**2))

    return out


def gram_with_RQ_kernel(sample_1, sample_2, alpha, l):
    """ calculate the gram matrix for RQ kernel between two sets of samples

    k^{rq}_{\alpha} (x, y) = 
        (1 + \frac{\|x-y\|^2}{2\alpha})^{-\alpha}
    
    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
        alpha: the parameter in RQ kernel
    
    Returns:
        the gram matrix (torch tensor, [n_1, n_2])
    """

    distances_squared = pdist_squared(sample_1, sample_2)

    return torch.pow(1 + distances_squared/(2*alpha*l**2), -alpha)

def MMD2_rbf_b(h, y, alpha, l):
    """ finite-sample biased estimate for squared MMD with Guassian kernel, aka RBF kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]

    out += (gram_with_RBF_kernel(h_1, h_1, l).mean() 
            - 2 * gram_with_RBF_kernel(h_1, h_2, l).mean() 
            + gram_with_RBF_kernel(h_2, h_2, l).mean())

    return out

def MMD2_rq_b(h, y, alpha, l):
    """ finite-sample biased estimate for squared MMD with rational quadratic kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]

    out = (gram_with_RQ_kernel(h_1, h_1, alpha, l).mean() 
                - 2 * gram_with_RQ_kernel(h_1, h_2, alpha, l).mean() 
                + gram_with_RQ_kernel(h_2, h_2, alpha, l).mean())
    return out

def MMD2_rq_u(h, y, alpha=1, l2=1):
    """ finite-sample unbiased estimate for squared MMD with rational quadratic kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]
    n_1 = h_1.shape[0]
    n_2 = h_2.shape[0]

    pd_12 = torch.cdist(h_1.unsqueeze(0),h_2.unsqueeze(0)).squeeze().reshape(-1)
    pd_11 = torch.pow(torch.nn.functional.pdist(h_1), 2)
    pd_22 = torch.pow(torch.nn.functional.pdist(h_2), 2)

    k_11 = torch.pow(1 + pd_11/(2 * alpha * l2), -alpha)
    k_22 = torch.pow(1 + pd_22/(2 * alpha * l2), -alpha)
    k_12 = torch.pow(1 + pd_12/(2 * alpha * l2), -alpha)
    out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                    - 2 * k_12.sum() / (n_1 * n_2)
                    + (k_22.sum() * 2)/(n_2*(n_2-1)))
    return out


def rq_from_k(l2, pd_11, pd_12, pd_22, alpha=1):
    k_11 = torch.pow(1 + pd_11/(2 * alpha * l2), -alpha)
    k_22 = torch.pow(1 + pd_22/(2 * alpha * l2), -alpha)
    k_12 = torch.pow(1 + pd_12/(2 * alpha * l2), -alpha)
    out = ((k_11.sum() - torch.trace(k_11)) / (n_1*(n_1-1))
                - 2 * k_12.sum() / (n_1 * n_2)
                + (k_22.sum() - torch.trace(k_22))/(n_2*(n_2-1)))
    return out

def MMD2_rbf_u(h, y, alpha=1, l2=1):
    """ finite-sample unbiased estimate for squared MMD with Guassian kernel, aka RBF kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]
    n_1 = int((y==0).sum())
    n_2 = int((y==1).sum())
    h = torch.cat([h_1, h_2], dim=0)

    pd_12 = torch.cdist(h_1.unsqueeze(0),h_2.unsqueeze(0)).squeeze().reshape(-1)
    pd_11 = torch.pow(torch.nn.functional.pdist(h_1), 2)
    pd_22 = torch.pow(torch.nn.functional.pdist(h_2), 2)
    if l2==0:
        out = 0
        for l2n in [1/16, 1/4, 1]:
            k_11 = torch.exp( - pd_11/(2 * l2n) )
            k_22 = torch.exp( - pd_22/(2 * l2n) )
            k_12 = torch.exp( - pd_12/(2 * l2n) )
            out += ((k_11.sum() * 2) / (n_1*(n_1-1))
                            - 2 * k_12.sum() / (n_1 * n_2)
                            + (k_22.sum() * 2)/(n_2*(n_2-1)))
        out = out / 3.0
    else:
        k_11 = torch.exp( - pd_11/(2 * l2) )
        k_22 = torch.exp( - pd_22/(2 * l2) )
        k_12 = torch.exp( - pd_12/(2 * l2) )
        out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))

    return out, l2


def MMD2MO_rq_u(h, y, alpha=1, l=1):
    """ finite-sample unbiased estimate for squared MMD with rational quadratic kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]
    n_1 = int((y==0).sum())
    n_2 = int((y==1).sum())
    h = torch.cat([h_1, h_2], dim=0)

    pd_12 = torch.cdist(h_1.unsqueeze(0),h_2.unsqueeze(0)).squeeze().reshape(-1)
    pd_11 = torch.pow(torch.nn.functional.pdist(h_1), 2)
    pd_22 = torch.pow(torch.nn.functional.pdist(h_2), 2)
    pd_12_op = pd_12.clone().detach()
    pd_11_op = pd_11.clone().detach()
    pd_22_op = pd_22.clone().detach()

    pd_all = torch.cat([pd_11, pd_12,pd_22],0)
    m = torch.median(pd_all).item()
    low = torch.min(pd_all).item()
    up = torch.max(pd_all).item()

    def fc_BFGS(logl2):
        with torch.no_grad():
            k_11 = torch.pow(1 + pd_11_op/(2 * alpha * math.e ** logl2[0]), -alpha)
            k_22 = torch.pow(1 + pd_22_op/(2 * alpha * math.e ** logl2[0]), -alpha)
            k_12 = torch.pow(1 + pd_12_op/(2 * alpha * math.e ** logl2[0]), -alpha)
            out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))
        return  - out.item()
    
    def fc_BFGS_der(logl2):
        with torch.no_grad():
            k_11 = math.e ** (- logl2[0]) / 2.0 * torch.mul(pd_11_op, torch.pow(1 + pd_11_op / (2 * alpha * math.e ** logl2[0]), -alpha-1))
            k_22 = math.e ** (- logl2[0]) / 2.0 * torch.mul(pd_22_op, torch.pow(1 + pd_22_op / (2 * alpha * math.e ** logl2[0]), -alpha-1))
            k_12 = math.e ** (- logl2[0]) / 2.0 * torch.mul(pd_12_op, torch.pow(1 + pd_12_op / (2 * alpha * math.e ** logl2[0]), -alpha-1))
            out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))
        return  np.array([- out.item()])

    # l2op = optimize.brent(fc_brent, brack=(low/10,m/2,up*10), maxiter=100)
    logl2op = optimize.minimize(fc_BFGS, np.array([math.log(m/2)]), method='BFGS', jac=fc_BFGS_der, tol=1e-5, options={'maxiter':20})
    if logl2op['success']:
        l2 = math.e ** logl2op['x'][0]
    else:
        l2 = m / 2

    k_11 = torch.pow(1 + pd_11/(2 * alpha * l2), -alpha)
    k_22 = torch.pow(1 + pd_22/(2 * alpha * l2), -alpha)
    k_12 = torch.pow(1 + pd_12/(2 * alpha * l2), -alpha)
    out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                    - 2 * k_12.sum() / (n_1 * n_2)
                    + (k_22.sum() * 2)/(n_2*(n_2-1)))
    return out, l2, math.e ** logl2op['x'][0], m/2


def MMD2MOA_rq_u(h, y, alpha=1, l=1):
    """ finite-sample unbiased estimate for squared MMD with rational quadratic kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]
    n_1 = int((y==0).sum())
    n_2 = int((y==1).sum())
    h = torch.cat([h_1, h_2], dim=0)

    pd_12 = torch.cdist(h_1.unsqueeze(0),h_2.unsqueeze(0)).squeeze().reshape(-1)
    pd_11 = torch.pow(torch.nn.functional.pdist(h_1), 2)
    pd_22 = torch.pow(torch.nn.functional.pdist(h_2), 2)
    pd_12_op = pd_12.clone().detach()
    pd_11_op = pd_11.clone().detach()
    pd_22_op = pd_22.clone().detach()

    pd_all = torch.cat([pd_11, pd_12,pd_22],0)
    m = torch.median(pd_all).item()
    low = torch.min(pd_all).item()
    up = torch.max(pd_all).item()

    def fc_BFGS(input):
        """
        x = log l2
        y = log alpha
        """
        x = input[0]
        y = input[1]
        t1 = 2 * math.e ** (x+y)
        t2 = math.e ** y
        with torch.no_grad():
            k_11 = torch.pow(1 + pd_11_op/t1, - t2)
            k_22 = torch.pow(1 + pd_22_op/t1, - t2)
            k_12 = torch.pow(1 + pd_12_op/t1, - t2)
            out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))
        return  - out.item()
    
    def fc_BFGS_der(input):
        """
        x = log l2
        y = log alpha
        """
        x = input[0]
        y = input[1]
        t1 = 2 * math.e ** (x+y)
        t2 = math.e ** y
        t3 = math.e ** (-x)
        with torch.no_grad():
            k_11 = t3 / 2.0 * torch.mul(pd_11_op, torch.pow(1 + pd_11_op / t1, -t2-1))
            k_22 = t3 / 2.0 * torch.mul(pd_22_op, torch.pow(1 + pd_22_op / t1, -t2-1))
            k_12 = t3 / 2.0 * torch.mul(pd_12_op, torch.pow(1 + pd_12_op / t1, -t2-1))
            grad_x = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))
            
            mat_11 = pd_11_op / t1 + 1
            mat_22 = pd_22_op / t1 + 1
            mat_12 = pd_12_op / t1 + 1
            k_11 = ( t3 / 2.0 * torch.mul(pd_11_op, 1 / mat_11) - t2 * torch.log(mat_11) ) * torch.pow(mat_11, -t2)
            k_22 = ( t3 / 2.0 * torch.mul(pd_22_op, 1 / mat_22) - t2 * torch.log(mat_22) ) * torch.pow(mat_22, -t2)
            k_12 = ( t3 / 2.0 * torch.mul(pd_12_op, 1 / mat_12) - t2 * torch.log(mat_12) ) * torch.pow(mat_12, -t2)
            grad_y = ((k_11.sum() * 2) / (n_1*(n_1-1))
                        - 2 * k_12.sum() / (n_1 * n_2)
                        + (k_22.sum() * 2)/(n_2*(n_2-1)))
        return  np.array([- grad_x.item(), - grad_y.item()])

    # l2op = optimize.brent(fc_brent, brack=(low/10,m/2,up*10), maxiter=100)
    opout = optimize.minimize(fc_BFGS, np.array([math.log(m/2), 0]), method='BFGS', jac=fc_BFGS_der, tol=1e-5, options={'maxiter':20})
    if opout['success']:
        l2 = math.e ** opout.x[0]
        alpha = math.e ** opout.x[1]
    else:
        l2 = m / 2
        alpha = 1

    k_11 = torch.pow(1 + pd_11/(2 * alpha * l2), -alpha)
    k_22 = torch.pow(1 + pd_22/(2 * alpha * l2), -alpha)
    k_12 = torch.pow(1 + pd_12/(2 * alpha * l2), -alpha)
    out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                    - 2 * k_12.sum() / (n_1 * n_2)
                    + (k_22.sum() * 2)/(n_2*(n_2-1)))
    return out, l2, math.e ** opout.x[0], m/2, alpha, math.e ** opout.x[1]

def MMD2_rq_u_all(h, y, alphas, l):
    alphas = [0.25, 0.5, 1, 2, 4]
    l = 1

    out = 0
    for alpha in alphas:
        out += MMD2_rq_u(h, y, alpha, l)
    out = out / len(alphas)
    return out