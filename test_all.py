
from financial_sentiment_analysis import (LMNDataReader, Tseries, lagger, 
                                            matrix_append, LMNDataReader2) 
import numpy as np
import datetime 

"""
def test_LMN_consistncy():
    url = 'data_for_financial_sentiment_paper.zip'
    data = sorted([i for i in LMNDataReader(url).nt_data()])
    data2 = sorted([i for i in LMNDataReader2(url).nt_data()])
    print(data[-3:])
    print(data2[-3:])
    
test_LMN_consistncy()
"""

def test_matrix_append():
    X = np.ones((4, 10))
    Y = np.zeros((5, 10))
    new = matrix_append(X, Y)
    assert new.shape == (9, 10) and sum(sum(new)) == 40

    
def test_lagger():
    x = range(10)
    Xt = lagger(x, 5, keep0=True)
    Xf = lagger(x, 5, keep0=False)
    
    test_arr = np.array([1, 2, 3, 4, 5])
    assert Xt.shape == (6, 10) and sum(Xt.T[0]-np.append(0, test_arr)) == 0 
    assert Xf.shape == (5, 10) and sum(Xf.T[0]-test_arr) == 0 


def test_LMNDataReader():
    dr = LMNDataReader('data_for_financial_sentiment_paper.zip')
    c = []
    for i in dr.nt_data():
        if i[1] == 'lexisnexis_AAPL': c.append(i[2])
        #assert type(i[0]) == datetime.date and type(i[1]) == str and type(i[2]) == float 
    print np.mean(c), np.median(c), np.max(c)
test_LMNDataReader()

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

