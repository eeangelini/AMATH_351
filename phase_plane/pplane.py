import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
import numdifftools as nd

def get_trajectories(mySystem,tVec,ICs):
    trajectory = {}
    for j,ic in enumerate(ICs):
        trajectory[j] = solve_ivp(mySystem,y0=ic,t_span=(tVec.min(),tVec.max()),t_eval=tVec).y
    return trajectory

def plot_trajectories(trajectory,ICs):
    for j,ic in enumerate(ICs):
        plt.plot(ic[0],ic[1],'b.',markersize=10)
        plt.plot(trajectory[j][0,:],trajectory[j][1,:], 'b-',linewidth=2.5)

def plot_components(tVec,trajectory,ICs):
    fig, ax = plt.subplots(1,2,figsize=(20,9))
    for j,ic in enumerate(ICs):
        L = ax[0].plot(tVec,trajectory[j][0,:],linewidth=2.5, label=ic)
        ax[1].plot(tVec,trajectory[j][1,:],linewidth=2.5, color=L[0].get_color(),label=ic)
    ax[0].set_xlabel(r"$t$",fontsize=16)
    ax[0].set_ylabel(r"$x_1(t)$",fontsize=16)
    ax[0].tick_params(labelsize=13)
    ax[0].set_title(r"Trajectories of $x_1$",fontsize=17)
    ax[0].legend(title='Initial conditions',title_fontsize=13,fontsize=12)
    ax[0].set_ylim([trajectory.min()-0.05,trajectory.max()+0.05])

    ax[1].set_xlabel(r"$t$",fontsize=16)
    ax[1].set_ylabel(r"$x_2(t)$",fontsize=16)
    ax[1].tick_params(labelsize=13)
    ax[1].set_title(r"Trajectories of $x_2$",fontsize=17)
    ax[1].legend(title='Initial conditions',title_fontsize=13,fontsize=12)
    ax[1].set_ylim([trajectory.min()-0.05,trajectory.max()+0.05])
    return fig,ax
    
def findroot(func, init): 
    """ Find root of nonlinear equation f(x)=0
    Args:
        - the system (function),
        - the initial values (list or np.array)

    return: roots of f(x) (np.array) 
            if the numerical method converge (else, return nan)
    """
    sol, info, convergence, sms = fsolve(func, init, full_output=1)
    if convergence == 1:
        return sol
    return np.array([np.nan]*len(init))

def get_fps(myFlow,ICs):
    '''Return the list of unique fixed points of the system x' = myFlow(x) starting around ICs
    '''
    fps = [] 
    # find each of the fixed points near the starting points numerically using the function findroot
    roots = [findroot(myFlow, ic) for ic in ICs]
    # Only keep unique fixed points and throw away 'nan' entries (findroot did not converge)
    for r in roots:
        if (not any(np.isnan(r)) and not any([all(np.isclose(r, x)) for x in fps])):
            fps.append(r)
    return fps

def find_stability(J):
    """ Determines stability of a fixed point given its associated 2x2 Jacobian matrix. 
    Args:
        J (np.array 2x2): the Jacobian matrix at the fixed point.
    Return:
        (string) classification of equilibrium point 
    """
    detJ = np.linalg.det(J)
    trJ = np.trace(J)
    if np.isclose(trJ,0) and detJ>0:
        nature = "Center"
    elif detJ < 0:
        nature = "Saddle point"
    else:
        nature = "Stable" if trJ < 0 else "Unstable" 
        nature += " spiral" if (trJ**2 < 4*detJ) else " node"
    return nature
        
def plot_null(f1,f2,x1,x2,params):
    X1,X2 = np.meshgrid(x1,x2)
    plt.contour(X1,X2,f1(X1,X2,**params),[0],colors='black', linestyles='dashed',linewidths=2)
    plt.contour(X1,X2,f2(X1,X2,**params),[0],colors='black', linestyles='dashed',linewidths=2)

def plot_flow(myFlow,x1,x2,numGrid = 15):
    X1,X2 = np.meshgrid(np.linspace(x1.min(),x1.max(),numGrid),np.linspace(x2.min(),x2.max(),numGrid))
    f = myFlow([X1,X2])
    f = f/np.sqrt(f[0]**2 + f[1]**2) # normalize vectors
    plt.quiver(X1,X2,f[0],f[1],width=0.003,alpha=0.5)

def plot_portrait(f1,f2,x1,x2,tVec,ICs,params,nullclines=True):
    # define function x' = [f1(x),f2(x)] for ODE solver: needs to have t as variable
    def mySystem(t,x):
        return np.array([f1(x[0],x[1],**params),f2(x[0],x[1],**params)])

    # define function x' = [f1(x),f2(x)] for rest of code (does not need t as variable)
    def myFlow(x):
        return mySystem(0,x)

    fig = plt.figure(figsize=(7,6))
    
    if nullclines: # plot nullclines
        plot_null(f1,f2,x1,x2,params)

    # plot direction field given by myFlow
    plot_flow(myFlow,x1,x2)
    
    # plot trajectories with initial conditions ICs
    trajectory = get_trajectories(mySystem,tVec,ICs)
    plot_trajectories(trajectory,ICs)
    
    fps = get_fps(myFlow,ICs) # get fixed points
    if len(fps) > 0: # check if there are fixed points
        # define Jacobian as a function of x - for getting stability:
        flowJacobian = nd.Jacobian(myFlow)
        print('Fixed points:')
        for fp in fps:
            fpStability = find_stability(flowJacobian(fp))
            print('  • '+fpStability+" at x = (%5.3f,%5.3f)" % (fp[0],fp[1]))
            plt.plot(fp[0],fp[1],'r.',markersize=15)
    
    plt.xlabel(r"$x_1$",fontsize=14)
    plt.ylabel(r"$x_2$",fontsize=14)
    plt.xlim([x1.min(),x1.max()])
    plt.ylim([x2.min(),x2.max()])
    return fig