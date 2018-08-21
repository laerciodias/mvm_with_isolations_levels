'''
Estimation of critical q.
'''
import itertools as __itertools
import matplotlib.pyplot as __plt
import numpy as __np
import pandas as __pd


import tools as __tools






def estimate(network_sizes, phys_quant, max_half_bandwidth=10,
             min_half_bandwidth=2, do_plot=False
             ):
    
    '''
    This method estimates the critical q (q_c).
    
    Use: qc, qc_error, best_regs, binder = est.estimate(<parameters>)
    
    Parameters:
    
    - network_sizes: the network sizes used for estimate q_c.
    - phys_quant: a pandas dataframe containing the values of the binder cumulant
                (besides magnetization and susceptibility).
    - min_half_bandwidth: the minimum amount of values of q at each side 
                around the value for a q_c candidate. 
    - max_half_bandwidth: the maximum amount of values of q at each side 
                around the value for a q_c candidate.
    - do_plot: if True the method will plot the regressions results.
    '''
    
    # Create an empty dataframe for binder cumulant values
    binder = __pd.DataFrame(index=phys_quant.index.levels[1])
    
    # Set columns as values of N, index as values of q.
    for n in network_sizes:
        binder[n] = phys_quant.loc[n].u
    
    # Let q be a column in the dataframe.
    # It is easy to do regressions in this way.
    binder.reset_index(inplace=True)
    
    # find the q_c candidate, considering the smallest and the greatest
    # network sizes.
    n1 = min(network_sizes)
    n2 = max(network_sizes)
    qCand = find_qc_candidate(n1,n2,phys_quant)
    
    if qCand is None:
        raise Exception('Probably there is no critical point. Check the plots of the Binder cumulant.')
    
    # find the index in binder dataframe for q_c
    binder2 = binder[binder.q == qCand]
    i_qC = binder2.index[0]
    
    # Set limits in the q values interval to do regressions.
    _ = i_qC - binder.index[0]
    LI = max_half_bandwidth if _ >= max_half_bandwidth else _
    
    _ = binder.index[-1] - i_qC
    LF = max_half_bandwidth if _ >= max_half_bandwidth else _
    
    if LI < min_half_bandwidth or LF < min_half_bandwidth:
        msg = 'qCand = %f is very close to the extreme values of q.\n' % qCand
        msg += 'Run more simulations to have q in a broader interval.'
        raise Exception(msg)
    
    # Do regressions considering subsets of q values around qCand.
    regressionsDF = []
    
    for li in range(min_half_bandwidth, LI+1):
        for ls in range(min_half_bandwidth,LF+1):
            binder2 = binder.loc[(i_qC-li):(i_qC+ls)]
            iI = binder2.index[0]
            iF = binder2.index[-1]
            
            for n in network_sizes:
                reg = __tools.regression(binder2.q, binder2[n])
                regressionsDF.append([binder2.loc[iI]['q'],
                                      binder2.loc[iF]['q'],
                                      n,
                                      reg.params[1], # coefficient
                                      reg.params[0], # intercept
                                      reg.rsquared,
                                     ])
    
    regressionsDF = __pd.DataFrame(data=regressionsDF,
                 columns=['qi', 'qf', 'N', 'coef', 'interc', 'R2'])
    
    # The criterium to choose which q interval we will be use:
    # Look at the variance of R2 for the network sizes.
    # Pick the inteval for the best R2 at the network size with the greatest
    # variance. Reason: if the binder cumulant is not linear for the
    # largest interval of values of q, we will have more variance. The best
    # R2 for the network size with biggest variance in R2 will indicate the
    # region of values of q where the binder cumulant is a better fit to 
    # a linear function.
    var = []
    for n in network_sizes:
        var.append(regressionsDF[regressionsDF.N == n].R2.var())
    var = __np.array(var)
    
    n = network_sizes[__np.argmax(var)]
    
    i_bestReg = regressionsDF[regressionsDF.N == n].R2.idxmax()
    
    qi = regressionsDF.loc[i_bestReg].qi
    qf = regressionsDF.loc[i_bestReg].qf
    
    # Get the values of binder cumulant for the best regression interval 
    binder2 = binder[(binder.q >= qi) & (binder.q <= qf)]
    
    # Get the regresisons results for the best regression interval
    best_regs = regressionsDF[(regressionsDF.qi == qi) & 
                            (regressionsDF.qf == qf)]
    best_regs.set_index(['N'], inplace=True)
    

    # Calculate the intersections for each pair of fittings of the binder cumulant.
       
    intersections = []
    
    for n1,n2 in list(__itertools.combinations(network_sizes,2)):
        
        a1 = best_regs.loc[n1].interc
        b1 = best_regs.loc[n1].coef
        a2 = best_regs.loc[n2].interc
        b2 = best_regs.loc[n2].coef
        
        p1 = __np.array([a1,b1])
        p2 = __np.array([a2,b2])
        q_inters = __tools.intersection(p1,p2)[0]
        
        intersections.append([n1,n2,q_inters])
    
    intersections = __pd.DataFrame(data=intersections, 
                                 columns=['n1','n2','q_inters'])
    
    # Calculate the critical value of q and the error
    qc = intersections.q_inters.mean()
    qc_error = intersections.q_inters.std()
    
    # Plot the regression results
    if do_plot:
        
        __plt.figure(figsize=(8,8))
        for n in network_sizes:
            
            l = best_regs.loc[n]
            
            X = __np.linspace(l.qi,l.qf)
            Y = l.interc + X*l.coef
            
            __plt.plot(X, Y, label = 'N=%d' % n)
            __plt.scatter(binder2.q, binder2[n], 
                        label = 'N=%d (reg)' % n,
                        marker='.'
                    )
        
        y = __plt.ylim()
        
        __plt.plot([qc,qc], y, color='black', label=r'$q_{c}$', lw=0.5)
        __plt.fill_betweenx(y, qc-qc_error, qc+qc_error, 
                        color='black',
                        alpha=0.15,
                        label=r'Error bar for $q_{c}$'
                        )
        
        __plt.legend(loc=(1.05, 0))
        __plt.xlabel('$q$', fontsize=18)
        __plt.ylabel(r'$u=-\ln\hspace{0.2}(1-3U/2)$', fontsize=18) 
        __plt.title(r'Binder cumulant (around $q_{c}$)', fontsize=18)
        
        __plt.show()
        
        print()
        print()
        print('CRITICAL POINT')
        print()
        print('       q_c = %.6f ± %.6f' % (qc, qc_error))   
    
    
    # return relevant information
    return qc, qc_error, best_regs, binder2
    











def find_qc_candidate(n1, n2, phys_quant, verbose=False):
    '''
    This function looks for the first value of q after the intersection
    between the Binder cumulant function u1 and u2 
    for networks sizes n1 and n2 respectively. The parameter phys_quant
    is the pandas dataframe where we stored the values of the physical
    interesting quantities (magnetization, susceptibility and binder
    cumulant).
    
    Under the assumption that u is "well-behaved", such intersection
    is a good candidate to be the critical point. So, do a visual
    inspection on the plot of u before use this funcion.
    
    The idea is that the intersection is marked by the change of signal
    at the difference between u1 and u2.
    
    Returns such q value, case it exists. Else, return None.
    
    '''
    # Get the values for the binder cumulant for both network sizes.
    u1 = phys_quant.loc[n1].u
    u2 = phys_quant.loc[n2].u
    
    # Calculate the signal of the difference between u1 and u2.
    s = __np.sign(u1 - u2)
    
    try:
        # The index where the signal changes occurs.
        i = __np.where(s != s[s.index[0]])[0][0]
        
        # Return the corresponding value of q.
        return s.index[i]
    
    except IndexError:
        # There is no intersection between u1 and u2.
        if verbose:
            print('Intersection not found between u(n=%d) and u(n=%d)!' % (n1, n2))
