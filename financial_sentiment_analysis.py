
from pylab import figure, show
import numpy as np
import datetime
import statsmodels.api as sm
import zipfile

FILELIST = {
    'web1': ['websites_AAPL_IBM_MSFT_IBM.csv','websites_everything_else.csv'],
    'web2': ['just_social_media_AAPL_IBM_MSFT_VZ.csv','just_social_media_everything_else.csv'],
    'lexisnexis': ['everything_else.csv - results-20160926-191305.csv.csv','AAPL_MSFT_VZ_AAPL.csv - results-20160926-190556.csv.csv'],
    'webAll': ['SOCIAL_AAPL_IBM_MSFT_VZ.csv - results-20160927-183311.csv.csv','SOCIALeverything_else.csv - results-20160927-183703.csv.csv']
    }
    
FFLIST = { 
    '3 factor': 'F-F_Research_Data_Factors_daily.CSV',
    '5 factor': 'F-F_Research_Data_5_Factors_2x3_daily.CSV'
    }

STOCKPRICELIST = {
    'price': 'stock_data.csv',
    'volume': 'stock_volumes.csv',
    'volatility': 'volatility_processed.csv'
    }
    
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
    
class Tseries:
   
    def __init__(self, idx):
        self.tseries = { i: {} for i in idx }
    
    def add(self, iter, on_missing='add'):
        
        for idx, name, val in iter:            
            if idx in self.tseries:
                self.tseries[idx][name] = val

    @property
    def descriptives(self):
        x = [0, 10000]
        names = None
        for dte, v in self.tseries.items():
            T = len(v)
            if T > x[0]: 
                x[0] = T
                names = v.keys()
            if T < x[1]:
                x[1] = T
                
        return x, len(self.tseries), names
    
    @property 
    def last_vals(self):
        cols = self.descriptives[2]
        last_vals = { c: None for c in cols }
        for dte in self.tseries:
            for c in cols:
                x = self.tseries[dte].get(c)
                if x is None or x == 0:
                    continue
                    
                last_vals[c] = dte
        return last_vals   
    
    def dummy_vars(self, cond, name):
        for dte in self.tseries.keys():
            if cond(dte):
                self.tseries[dte][name] = 1
            else:
                self.tseries[dte][name] = 0
    
    def make_return_matrix(self):
        """
        Generates a dts x cols matrix and associated labels
        """
    
        dts = sorted(self.tseries.keys())
        cols = self.descriptives[2]
        
        matrix = np.zeros((len(dts), len(cols)))
        for n, dte in enumerate(dts):
            for k, col in enumerate(cols):
                matrix[n][k] = self.tseries[dte][col]
        
        self.matrix = matrix
        self.dates = dts
        self.labels = list(cols)
    
    def remove_null(self):
        ts = {}
        for dte in self.tseries:
            if len(self.tseries[dte]) > 0:
                ts[dte] = self.tseries[dte]
        self.tseries = ts
    
    def pad_missing(self):
        cols = self.descriptives[2]
        for dte in self.tseries:
            for c in cols:
                if c not in self.tseries[dte]:
                    self.tseries[dte][c] = 0
    
    def matrix_selector(self, label_selection):
        selection = []
        for i in label_selection:
            for n, col in enumerate(self.labels):
                if col == i:
                    selection.append(self.matrix.T[n])
        
        selection = np.array(selection).T
        return selection 
        
    
         
class LMNDataReader:
    
    def __init__(self, url):
        self.url = url
    
    def _date_converter(self, x):
        x = x.split()[0]
        
        if '-' in x:
            x = list(map(int, x.split('-')))
            x = datetime.date(x[0], x[1], x[2])
        elif '/' in x:
            x = list(map(int, x.split('/')))
            x = datetime.date(x[2], x[0], x[1])
        else:
            print('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            x = int(x)
        except:
            print('error trying to parse nt: {}'.format(x))
        
        return x
        
    def nt_data(self):
        """
        get data into a dict like
            data[source content][ticker] = [date, LMN words]
        """
            
        data = {}
        zf = zipfile.ZipFile(self.url)
        for k, names in FILELIST.items():
            data[k] = {}
            for fname in names:
                fcontent = zf.open(fname).readlines()
                for line in fcontent:
                    line = line.decode("utf-8")[:-1].split(',')

                    dte = self._date_converter(line[1])
                    nt = self._nt_converter(line[2])
                    
                    yield dte, "{}_{}".format(k, line[0]), nt
                    
class FFDataReader:
    
    def __init__(self, url):    
        self.url = url
    
    def _ff_date_converter(self, x):
        try: 
            x = datetime.date(int(x[:4]), int(x[4:6]), int(x[6:]))
        except Exception as E:
            print(repr(E))
        
        return x
        
    def FF_data(self, greater_than=20140000):
        """
        get data into a dict like
            data[ff model] = [date, factors...]
        """
    
        zf = zipfile.ZipFile(self.url)
         
        for k, fname in FFLIST.items():
            fcontent = zf.open(fname).readlines()
            for line in fcontent:
                line = line[:-2].split(b',')
                try:
                    assert int(line[0]) > greater_than
                except Exception as E:
                    continue
                
                dte = self._ff_date_converter(line[0])
                if k == '3 factor':
                    for n in range(1, 4):
                        yield dte, '{} f={}'.format(k, n), float(line[n])
                elif k == '5 factor':
                    for n in range(1, 6):
                        yield dte, '{} f={}'.format(k, n), float(line[n])

class StockPriceDataFeeder:
    def __init__(self, url):
        self.url = url
        
    def _date_converter(self, x):
        x = x.split(b'/')
        x = datetime.date(int(x[2]), int(x[1]), int(x[0]))
        return x
            
    def sp_data(self):
        zf = zipfile.ZipFile(self.url)
         
        for k, fname in STOCKPRICELIST.items(): 
            fcontent = zf.open(fname).readlines()
            tickers = fcontent[0][:-2].split(b',')[1:]
            tickers = [i.decode("utf-8") for i in tickers]
            for line in fcontent:
                line = line[:-2].split(b',')
                try:
                    dte = self._date_converter(line[0])
                except Exception as E:
                    print(repr(E))
                    continue
                    
                for n, ticker in enumerate(tickers):
                    yield dte, "{}_{}".format(k, ticker), float(line[n+1])                    

class MatrixOperations:
    def zscore(self, X):
        mu = np.mean(X, 0)
        sig = np.std(X, 0)
        X = (X-mu)/sig
        return X
        
    def diff(self, X, n):
        X = X.T
        X[n] = np.append(0, np.diff(X[n]))
        X = X.T
        return X 

    def var(self, X, n, lags):
        X = X.T
        X_ = np.zeros((X.shape[0]+len(lags), X.shape[1]))
        X_[:X.shape[0]] = X
        c = X.shape[0]
        for lag in lags:
            laggedX = X[n]
            laggedX = laggedX[:-lag]
            laggedX = np.append(np.zeros(lag), laggedX)
            X_[c] = laggedX
            c += 1
        X = X_.T
        return X
        
    def append(self, X1, X2):
        X = np.zeros((X1.shape[0]+X2.shape[0], X1.shape[1]))
        X[:X1.shape[0]] = X1
        X[X1.shape[0]:] = X2
        return X

def make_var(ticker, lags):
    mo = MatrixOperations()
    X = ts.matrix_selector(['price_{}'.format(ticker),
                                'lexisnexis_{}'.format(ticker),
                                'NWD', 'friday', 'january'
                                ])
    y = ts.matrix_selector(['price_{}'.format(ticker)])
    
    X = mo.diff(X, 0)[:-1]
    y = mo.diff(y, 0)[1:]
    X = mo.var(X, 1, range(1, lags))
    X = mo.var(X, 0, range(1, lags))
    # remove 'padded' var valuess
    X = X[lags:]
    y = y[lags:]
    X = mo.zscore(X)
    y = mo.zscore(y)
    return X, y

def pannal_var(tickers, lags):
    mo = MatrixOperations()
    X, y = make_var(tickers[0], lags)
    for i in tickers[1:]:
        X_, y_ = make_var(i, lags)
        X = mo.append(X, X_)
        y = mo.append(y, y_)
    
    print(X.shape, y.shape)
    res = sm.OLS(y, sm.tools.add_constant(X)).fit()
    print(res.summary())
        
    
if __name__ == '__main__':  

    #prep data
    min_date = datetime.date(2014, 1, 1)
    max_date = datetime.date(2015, 8, 23)
    url = 'data_for_financial_sentiment_paper.zip'

    ts = Tseries(date_range(min_date, max_date))
    
    #trading days first so we can exclude non trading days 
    ts.add(StockPriceDataFeeder(url).sp_data())
    ts.remove_null()
    
    ts.add(LMNDataReader(url).nt_data())
    ts.add(FFDataReader(url).FF_data())
     
    ts.remove_null()
    ts.pad_missing()       
    ts.dummy_vars(lambda x: x.weekday()==0, 'NWD')
    ts.dummy_vars(lambda x: x.weekday()==4, 'friday')
    ts.dummy_vars(lambda x: x.month==1, 'january')
    
    ts.make_return_matrix()
    
    # do regressions
    all_tickers = [i.split('_')[1] for i in ts.descriptives[2] if 'price' in i]    
    pannal_var(all_tickers, 5)
    
    
# regression accross all stocks
# VAR per stock 

    
"""


def get_crsp_data():
    with open('../Downloads/CRSP value-eighted index return.xlsx - WRDS.csv', 'rb') as inp:
        data = {}
        for line in csv.reader(inp):
            try:
                line[0] = map(int, [line[0][:4], line[0][4:6], line[0][6:]])
                line[0] = datetime.date(line[0][0], line[0][1], line[0][2])
                line[1] = float(line[1])
                data[line[0]] = { 'crsp': line[1] }
                
            except Exception as E:
                print repr(E)
                 
        return data
    


def var(x, y, tau, january, friday=None, single=False):
    X = []
    X_ = []
    for i in range(1, tau): X_.append(zs(x[tau+i:-(tau-i)]))
    X = X+X_[::-1]

    if single != True:
        X_ = []
        for i in range(1, tau): X_.append(zs(y[tau+i:-(tau-i)]))
        X = X+X_[::-1]

    X.append(january[tau*2:])
    if friday != None: X.append(friday[tau*2:])
    
    X = np.array(X).T
    mu, sig = [np.mean(y), np.std(y)]
    y = zs(y[tau+tau:])
    
    res = sm.OLS(y, sm.tools.tools.add_constant(X)).fit()
    if 1 == 0:
        print list(res.params)
        print list(res.pvalues)
        print res.rsquared_adj
    return [y, X, mu, sig]


def rolling_reg(x, y, tau, dts):
    X = []
    X_ = []
    for i in range(1, tau):
        X_.append(zs(x[tau+i:-(tau-i)]))
        
    X = X+X_[::-1]
    X_ = []
    for i in range(1, tau): X_.append(zs(y[tau+i:-(tau-i)]))
    X = X+X_[::-1]

    X.append(january[tau*2:])
    X.append(friday[tau*2:])

    X = np.array(X).T
    
    y = zs(y[tau+tau:])
    dts = dts[tau+tau:]


    results = [[],[],[],[]]
    sigdts = {}    
    for n, i in enumerate(y):
        if n < 250: continue
        
        res = sm.OLS(y[n-250:n+1], sm.tools.tools.add_constant(X[n-250:n+1])).fit()

        # is lag 1 significant 
        results[1].append(res.params[1])
        results[2].append(sum(res.params[1:6]))
        results[3].append(sum(res.params[2:6]))
        results[0].append(dts[n])

        if res.tvalues[1] < -1.96:
            sigdts = daterange(dts[n-250], dts[n], sigdts, 'lag1')
        if sum(res.params[1:6])/np.mean(res.params[1:6]*(1./res.tvalues[1:6])) < -1.96:
            sigdts = daterange(dts[n-250], dts[n], sigdts, 'pers')
        if sum(res.params[2:6])/np.mean(res.params[2:6]*(1./res.tvalues[2:6])) > 1.96:
            sigdts = daterange(dts[n-250], dts[n], sigdts, 'tran')


    rmkr_save, pmkr_save, tmkr_save = [1, 1, 1]
    rmkr, pmkr, tmkr = [1, 1, 1]
    rperiods, tperiods, pperiods  = [[[],[]],[[],[]],[[],[]]]
    #for i in dts[250:]:
    for i in dts:
        if i in sigdts:
            if 'lag1' in sigdts[i]: rmkr = 0
            else: rmkr = 1
            if 'lag1' in sigdts[i] and 'tran' in sigdts[i]: tmkr = 0
            else: tmkr = 1
            if 'lag1' in sigdts[i] and 'pers' in sigdts[i]: pmkr = 0
            else: pmkr = 1
            
        if rmkr != rmkr_save: rperiods[rmkr].append(i)
        if tmkr != tmkr_save: tperiods[tmkr].append(i)
        if pmkr != pmkr_save: pperiods[pmkr].append(i)

        rmkr_save, pmkr_save, tmkr_save = [rmkr, pmkr, tmkr]
        
    if rmkr == 0: rperiods[1].append(i)
    if tmkr == 0: tperiods[1].append(i)
    if pmkr == 0: pperiods[1].append(i)

    rperiods = np.array(rperiods).T
    tperiods = np.array(tperiods).T
    pperiods = np.array(pperiods).T

    print 'lag 1 significant', rperiods
    print 'transient periods', tperiods
    print 'persistent periods', pperiods
    if len(rperiods) > 0:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.plot(results[0], results[1], color='k', label='lag1')
        ax.plot(results[0], results[2], color='k', linestyle='--', label='sum 1:5')
        ax.plot(results[0], results[3], color='k', linestyle='-.', label='sum 2:5')
        for i in rperiods:
            ax.axvspan(i[0], i[1], color='k', alpha=0.2)
        #for i in pperiods:
        #    ax.axvspan(i[0], i[1], color='g', alpha=0.2)
        #for i in tperiods:
        #    ax.axvspan(i[0], i[1], color='r', alpha=0.2)
        ax.legend(loc='best')
        show()
        
def get_ff_regressions(dta, lmn, r, crsp):   
    ff_dts, ff_factors = get_ff_factors(dts, 0)
    ff5_dts, ff5_factors = get_ff_factors(dts, 1)
    port = []
    for n, i in enumerate(lmn.T):
        high = [n for n, x in enumerate(i) if x >= np.median(i)]
        low = [n for n, x in enumerate(i) if x < np.median(i)]
        try:
            port.append(((-sum(r[n+1][high]))+sum(r[n+1][low]))/len(i))
        except Exception as E:
            print repr(E)

    print ff_factors.shape, len(port)

    def cumsum(x):
        d = [0]
        for i in x: d.append(d[-1]+i)
        return d

    pprintres(sm.OLS(port, sm.tools.tools.add_constant(ff_factors)).fit())
    pprintres(sm.OLS(port, sm.tools.tools.add_constant(ff5_factors)).fit())

    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(cumsum(port))
    ax.plot(cumsum(crsp))
    show()


def check_vars(dts, price, lmn):
    fig = figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.plot(dts, price)
    ax1.plot(dts, lmn)
    show()

tickers_lmn, dts_lmn, lmn = get_lmn_data()
tickers_price, dts_prices, prices = get_stock_price_data2()
crsp_data = get_crsp_data()

data = map_data_together(crsp_data, [dts_prices, prices], 'r')
data = map_data_together(data, [dts_lmn, lmn], 'lmn')
dts, price, lmn, crsp, january, friday, NWK = prep_for_analysis(data, tickers_price, tickers_lmn)

#check_vars(dts, price, lmn)

r = np.diff(np.log(price).T).T
print r.shape

### prep the data
january = january[1:]
friday = friday[1:]
NWK = NWK[1:]
lmn = np.array(lmn[1:])
dts = dts[1:]
crsp = np.array(crsp[1:])
ar = (r.T-crsp)
lmn = lmn.T
r = r.T




for n, i in enumerate(ar):
    print tickers_price[n]
    rolling_reg(lmn[n], i, 6, dts)   

#get_ff_regressions(dts, lmn, r, crsp)
sys.exit()
### var models and pannal

y = []
X = []
for n, i in enumerate(r): # change to r for vol regs
    if tickers_price == 'PFE': continue
    #print '\n\n', tickers_price[n], '\n'
    #y_, X_, mu, sig = var(lmn[n], i, 6, january, friday)
    #y_, X_, mu, sig = var(i, lmn[n], 6, january, friday)

    y_, X_, mu, sig = var(lmn[n], i, 3, NWK, crsp, single=True)
    
    y = y+list(y_)
    X = X+list(X_)
    print mu, sig
    
y, X = [np.array(y), np.array(X)]
print y.shape, X.shape, mu, sig

res = sm.OLS(y, sm.tools.tools.add_co nstant(X)).fit()

print '\n\nPANEL\n\n'

pprintres(res)

print sum(res.params[1:6]), sum(res.params[2:6])
#print norm(sum(res.params[1:6])/np.mean(res.params[1:6]*(1./res.tvalues[1:6])))
#print norm(sum(res.params[2:6])/np.mean(res.params[2:6]*(1./res.tvalues[2:6])))

print res.summary()



"""








