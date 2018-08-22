'''
The goal of this class is provide general utility methods for other classes and notebooks.
'''
from scipy.interpolate import interp1d as __interp1d
from scipy.misc import derivative as __scipy_diff

import numpy as __np
import statsmodels.api as __sm





def derivative(x, y, interp_kind='cubic', dx=1e-8, num_multiply=4):
    '''
    Calculate numerical derivatives, using interpolation to improve results.
    
    Parameters:
    - num_multiply: if x is an array with n points, this funcion will return an array with aproximately n*num_multiply points.
                    To have more points is crucial to improving the accuracy of the interpolation.
                    
    Returns an expanded version of x, X, and the corresponding values of the derivative, dY.
    X and dY are numpy arrays.
    
    NOTE: This function will be not so good at the extremes of the interval, particularly for hard functions like oscillating functions.
    
    '''

    # Interpolate y as a function of x.
    f = __interp1d(x, y, kind=interp_kind)
    
    # Create a grid X with more points than x, but based on it.
    X = __np.linspace(min(x), max(x), num=num_multiply*(len(x)-1)+1)
    
    # The derivative cannot be calculated at extreme points.
    X = X[1:-1]
    
    # Use the derivative method from scipy to calculate the derivative at each xi in X.
    dY = __np.array([__scipy_diff(f, xi, dx=dx) for xi in X])
    
    return X, dY



def express_measure_with_error(measure, error, label=''):
    '''
    Parameters:
    - label: A label for the quantity expressed. 
            Example: 'M' for magnetization.
            If the label is an empty string (default value) 
            the method returns the measure and the error with the
            correct number of digits.
            Else it returns a formatted string.
    '''
    
    # I do the trick taking advantage of string representations.
    # I know, maybe this way is not bullet proof.
    s = '%.e' % error
    dec = int(__np.abs(float(s[s.find('e')+1:])))
    
    if error < 1:
        dec_format = '%s%df' % ('%.', dec)
        xx = (dec_format % measure, dec_format % error)
    else:
        xx = (int(__np.round(measure, -dec)), 
              int(__np.round(error, -dec)))
    
    if label == '':
        return float(xx[0]), float(xx[1])
    else:
        return '%s = %s ± %s' % (label, xx[0],xx[1])






def intersection(p1, p2):
    '''
    Calculates the intersection between the lines
    
    y = a1 + b1*x and y = a2 + b2*x
    
    where p1 = array([a1,b1]) and p2 = array([a2,b2])
    '''
    a1 = p1[0]
    b1 = p1[1]
    
    a2 = p2[0]
    b2 = p2[1]
    
    if b1 == b2:
        raise Exception('Parallel lines')
    
    d = (b1-b2)
    x = 1.*(a2-a1)/d
    y = 1.*(b1*a2-b2*a1)/d
    
    return x,y



def regression(X, Y):
    '''
    In general I am interested in results.params and results.rsquared
    '''
    X = __sm.add_constant(X)

    model = __sm.OLS(Y,X)
    results = model.fit()
    
    return results



def regression_power_law(x, y):
    '''
    In general I am interested in results.params and results.rsquared
    '''
    Y = __np.log10(y)
    X = __np.log10(x)
    X = __sm.add_constant(X)

    model = __sm.OLS(Y,X)
    results = model.fit()
    
    return results



