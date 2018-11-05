
from financial_sentiment_analysis import (LMNDataReader, Tseries, lagger, 
                                            matrix_append) 
import numpy as np
import datetime 

def test_matrix_append():
    X = np.ones((4, 10))
    Y = np.zeros((5, 10))
    new = matrix_append(X, Y)
    assert new.shape == (9, 10) and sum(sum(new)) == 40
    
def test_lagger():
    x = range(10)
    Xt = lagger(x, 5, keep0=True)
    Xf = lagger(x, 5, keep0=False)
    
    test_arr = np.array([9, 8, 7, 6, 5, 4])

    assert Xt.shape == (6, 10) and sum(Xt.T[-1]-test_arr) == 0 
    assert Xf.shape == (5, 10) and sum(Xf.T[-1]-test_arr[1:]) == 0 

def test_LMNDataReader():
    dr = LMNDataReader('data_for_financial_sentiment_paper.zip')
    for i in dr.nt_data():
        assert type(i[0]) == datetime.date and type(i[1]) == str and type(i[2]) == float 
        
def test_select():
    def _test_iter():
        dte = datetime.date(2000, 1, 1)
        for i in range(20):
            yield dte, 'test', i
            dte = dte+datetime.timedelta(-1)
    
    idx = [i[0] for i in _test_iter()] 
    ts = Tseries(idx)
    ts.add(_test_iter())
    dts, vals = ts.select('test')
    for i in range(1, len(dts)):
        assert ((dts[i]-dts[i-1])).days == 1
