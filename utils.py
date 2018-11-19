
import datetime
import numpy as np

def date_range(min_date, max_date, weekdays=False):
    dts = []
    dte = min_date
    while dte <= max_date:
        if weekdays is False and dte.weekday() > 4:
            pass
        else:
            dts.append(dte)
        dte += datetime.timedelta(1)
    
    return dts
    
def zscore(x):
    x = (x-np.mean(x))/np.std(x)
    return x
    
def matrix_append(X, Y):
    """
    assumes 2 matrices NxM where Xm=Ym and you are appending over the N
    axis  
    """
    new = np.zeros((X.shape[0]+Y.shape[0], X.shape[1]))
    new[:X.shape[0]] = X
    new[X.shape[0]:] = Y
    return new