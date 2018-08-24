from scipy.interpolate import interp1d as __interp1d

import matplotlib.pyplot as __plt
import numpy as __np

import tools as __tools



'''
The goal of this class is to provide methods to estimate all the critical exponents.

'''



def estimate_D(beta_nu, beta_nu_err, gamma_nu, gamma_nu_err):
    '''
    Given some exponents, calculates the hyper-scale relation (with error).
    '''
    return 2*beta_nu + gamma_nu, 2*beta_nu_err + gamma_nu_err



def estimate_exponent(exponent, qc, network_sizes, df, 
                      do_plot=False, scale_var='N'):
    
    '''
    Estimates the exponents 'beta_nu', 'gamma_nu', 'inv_nu' (one per method's  call).
    
    Parameters:
    - exponent: {'beta_nu', 'gamma_nu', 'inv_nu'}
    - qc: the critical value for q.
    - network_sizes: A list of values of the network sizes.
    - df - a pandas data frame containing the physical quantities.
    - do_plot: if 'True' a plot will be drawn.
    - scale_var: {'L', 'N'}
    
    Use:
        exp, exp_err, y_til_0, y_til_0_err, reg_rsquared = estimate_exponent(<parameters>)
        
    '''
    
    if scale_var == 'L':
        # Linear size of the networks
        D = __np.array(network_sizes)**0.5
    elif scale_var == 'N':
        D = __np.array(network_sizes)
    else:
        raise NotImplementedError("Parameter 'scale_var' is not valid.")
        
    
    # Leave just 'N' in the index.
    df = df.reset_index(['q']).copy()
    
    
    # Set values to use along the method based on the 'exponent' parameter.
    if exponent == 'beta_nu':
        y = 'M'
        signal = -1
        yLabel = 'M_{N}'
        expLabel = r'$\beta/\nu$'
        y_til_label = r'$\widetilde{M}(0)$'
    elif exponent == 'gamma_nu':
        y = 'X'
        signal = 1
        yLabel = r'\chi_{N}'
        expLabel = r'$\gamma/\nu$'
        y_til_label = r'$\widetilde{\chi}(0)$'
    elif exponent == 'inv_nu':
        y = "U"
        signal = 1
        yLabel = r"U'_{N}"
        expLabel = r'$1/\nu$'
        y_til_label = r"$\widetilde{U'}(0)$"
    else:
        raise NotImplementedError("Parameter 'exponent' not valid.")
        
    
    # Calculating the y function on q_c (with the error for the Magnetization) by linear interpolation.
    yc = []
    yc_err = []
    for n in network_sizes:
        # Calculate the interpolators.
        if exponent == 'inv_nu':
            f = __tools.derivative(df.loc[n]['q'], df.loc[n][y])
        else:
            f = __interp1d(df.loc[n]['q'], df.loc[n][y])

            
        if exponent == 'beta_nu':
            f_err = __interp1d(df.loc[n].q, df.loc[n]['M_err'], 
                             kind='cubic')
        
        # Calculate the y function on qc.
        if exponent == 'inv_nu':
            yc.append(float(__np.abs(f(qc))))
        else:
            yc.append(float(f(qc)))
        
        if exponent == 'beta_nu':
            yc_err.append(float(f_err(qc)))
    
    yc = __np.array(yc)
    if exponent == 'beta_nu':
        yc_err = __np.array(yc_err)
    
    
    # Do a power law fitting.
    reg = __tools.regression_power_law(D,yc)
    exp = signal*reg.params[1]
    exp_err = __np.sqrt(reg.cov_params()[1][1])
    y_til_0 = 10**reg.params[0]
    y_til_0_err = y_til_0*__np.log(10)*__np.sqrt(reg.cov_params()[0][0])
    
    
    # Do a plot, if requested.
    if do_plot:
        X = __np.linspace(D[0],D[-1])
        A = y_til_0
        B = signal*exp
        Y = A*X**B
    
        __plt.figure(figsize=(8,6))
        
        if exponent == 'beta_nu':
            __plt.errorbar(D, yc, yc_err, lw=0.5, marker='.', ls='', 
                         capsize=4)
        else: 
            __plt.plot(D, yc, yc_err, lw=0.5, marker='.', ls='')
           
        __plt.plot(X, Y, lw=0.5)
        __plt.xscale('log')
        __plt.yscale('log')
        __plt.xlabel(scale_var, fontsize=16)
        __plt.ylabel(r"$%s(q_{c})$" % (yLabel), fontsize=16)
        __plt.title('%s   %s   (R²=%.4f)' % (
            __tools.express_measure_with_error(exp, exp_err, 
                               label = r'%s' % expLabel), 
            __tools.express_measure_with_error(y_til_0, y_til_0_err, 
                               label = r'%s' % y_til_label),
            reg.rsquared), 
                  fontsize=12
                 )
        
        __plt.xscale('log')
        __plt.yscale('log')
        

    return exp, exp_err, y_til_0, y_til_0_err, reg.rsquared 



def data_collapse(qc, beta_nu, gamma_nu, inv_nu,
                  network_sizes, df_quantities, 
                  scale_var='N', quantities_labels = ['M', 'X']):
    
    '''
    Do the data collapse plots for the Magnetization ('M') and the Susceptibility ('X').
    
    Magnetization: M_N(q)*N^{\beta/\nu} as a function of |q-q_{c}|*N^{1/\nu}.
    
    Susceptibility: X_N(q)*N^{\beta/\nu} as a function of (q-q_{c})*N^{1/\nu}.
    
    Parameters:
    - exponent: {}
    - qc: the critical value for q.
    - beta_nu, gamma_nu, inv_nu: critical exponents already calculated.
    - network_sizes: A list of values of the network sizes.
    - df_quantities - a pandas data frame containing the physical quantities.
    - scale_var: {'L', 'N'}.
    - quantities_labels = ['M'], ['X] or ['M', 'X']. 
    
    This method does not have a return.
        
    '''
    
    # Reset the index of physical quantities data frame.
    df_quantities = df_quantities.reset_index().copy()
    
    if scale_var == 'L':
        # Linear size of the networks.
        D = df_quantities['N']**0.5
    elif scale_var == 'N':
        D = df_quantities['N']
    else:
        raise NotImplementedError("Parameter 'scale_var' is not valid.")
    
    symbols = ['<','>', 'D', 'h', 'v']
    cs = ['black', 'blue','orange','green','red']
    
    for q_label in quantities_labels:
        # y_tilde represents the "quantity tilde" of interest.
        if q_label == 'M':
            df_quantities['y_tilde'] = df_quantities['M']*D**(beta_nu)
            df_quantities['x'] = (df_quantities['q'] - qc).abs()*D**(inv_nu)
            xlabel = r'$|q-q_c|%s^{1/\nu}$' % (scale_var)
            ylabel = r"$M_{%s}(q)%s^{\beta/\nu}$" % (scale_var,scale_var)
        
        if q_label == 'X':
            df_quantities['y_tilde'] = df_quantities['X']*D**(-gamma_nu)
            df_quantities['x'] = (df_quantities['q'] - qc)*D**(inv_nu)
            xlabel = r'$(q-q_c)%s^{1/\nu}$' % (scale_var)
            ylabel = r"$\chi_{%s}(q)%s^{\beta/\nu}$" % (scale_var,scale_var)
        
        __plt.figure(figsize=(7,5))
        for i in range(len(network_sizes)):
            df_quantities_slice = df_quantities[df_quantities['N'] == network_sizes[i]]
            label = network_sizes[i] if scale_var == 'N' else network_sizes[i]**0.5
            __plt.scatter(df_quantities_slice['x'], 
                        df_quantities_slice['y_tilde'], s = 20,
                        label = '%s = %d' % (scale_var, label), 
                        marker=symbols[i], facecolors="None", 
                        color=cs[i], lw=0.6)
        
        __plt.xlim(df_quantities['x'].min(), df_quantities['x'].max())
        __plt.ylim(df_quantities['y_tilde'].min(),
                df_quantities['y_tilde'].max())
        
        __plt.xlabel(xlabel, fontsize=18)
        __plt.ylabel(ylabel, fontsize=18)
    
        __plt.legend(loc=(1.05,0))
        
        if q_label == 'M':
            __plt.xscale('log')



