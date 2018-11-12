
"""
Excluding PFE because there is an anomally in the data 
need to replicate origional results 
    - there appears to be an issue with LMNDataReader2 focus on the origional
        for now
    - **use returns not adj returns in the origional
    - the word counts between the 2 LMNDataReaders are completely different 
        this appears to be a fault with the data 
        
        NEED TO REPLICATE IN BIGQUERY 
        - tried with lexisnexis gt3 mentions and is quite different 
        - so are the results with raw_data 
        - also the BA result is lower max suggesting it isnt caused by 
            filtering 
        - AAPL max is way higher 
        - *the new deduped and stop word filtered sets appear to be working* 
        
        - need to get a handle, can w confirm avg messages and total words for 
            one dataset and then work from there?
            last query is closest to repolicating descriptives, need to confirm
            with total words 
            
    STATSMODELS RESULTS are in the intuative order reading left to right on 
        the columns of the endog matrix 
        
    - its interesting the deduped data works and the undeduped data does not 
    
    - next steps
        - replicate all results
        - check methodology against paper and write docstrings detailing 
            implementation details in line with this work 
            
        - need to change logging to be more graceful when set dynamically 
"""

from pylab import figure, show
import numpy as np
import datetime
import statsmodels.api as sm
import zipfile
import logging
import sys 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__main__')

FILELIST = {
    'web1': ['websites_AAPL_IBM_MSFT_IBM.csv','websites_everything_else.csv'],
    'web2': ['just_social_media_AAPL_IBM_MSFT_VZ.csv','just_social_media_everything_else.csv'],
    'lexisnexis': ['everything_else.csv - results-20160926-191305.csv.csv','AAPL_MSFT_VZ_AAPL.csv - results-20160926-190556.csv.csv'],
    'web': ['SOCIAL_AAPL_IBM_MSFT_VZ.csv - results-20160927-183311.csv.csv','SOCIALeverything_else.csv - results-20160927-183703.csv.csv']
    }

FILELIST3 = {
    'lexisnexis': 'lexisnexis_word_counts_deduped.csv',
    'web': 'web_word_counts_deduped.csv',
    'web1': 'web1_word_counts_deduped.csv',
    'web2': 'web2_word_counts_deduped.csv'
    }
    
FILELIST2 = {
    'web': 'web',
    'web1': 'web1',
    'web2': 'web2',
    'lexisnexis': 'lexisnexis'
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
    def tickers(self):
        tickers = [i.split('_')[1] for i in self.descriptives[2] 
                    if 'price' in i]
        return tickers 
    
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
    
    def remove_missing(self):
        new_tseries = {}
        cols = self.descriptives[2]
        for dte in self.tseries:
            marker = True
            for c in cols:
                if c not in self.tseries[dte]:
                    marker = False
            if marker:
                new_tseries[dte] = self.tseries[dte]
        
        self.tseries = new_tseries
                    
                    
    def select(self, ticker):
        ts = sorted([(k, i[ticker]) for k, i in self.tseries.items()])
        dts, vals = zip(*ts)
        return np.array(dts), np.array(vals)

    
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
            logger.warn('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            x = float(x)
        except:
            logger.warn('error trying to parse nt: {}'.format(x))
        
        return x
        
    def nt_data(self):
        """
        get data into a dict like
            data[source content][ticker] = [date, LMN words]
        """
            
        data = {}
        zf = zipfile.ZipFile(self.url)
        for k, fnames in FILELIST.items():
            for fname in fnames:
                data[k] = {}
                fcontent = zf.open(fname).readlines()
                for line in fcontent[1:]:
                    line = line.decode("utf-8")[:-1].split(',')
                    

                    dte = self._date_converter(line[1])
                    nt = self._nt_converter(line[-1])
                    
                    yield dte, "{}_{}".format(k, line[0]), nt
               
class LMNDataReader2:
    
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
            logger.warn('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, T, x):
        
        try:
            #x = 100*(float(x)/float(T))
            x = float(x)
        except:
            logger.warnv('error trying to parse nt: {}'.format(x))
        
        return x
        
    def nt_data(self):
        """
        get data into a dict like
            data[source content][ticker] = [date, LMN words]
        """
            
        data = {}
        zf = zipfile.ZipFile(self.url)
        for k, fname in FILELIST2.items():
            data[k] = {}
            fcontent = zf.open(fname).readlines()
            for line in fcontent[1:]:
                line = line.decode("utf-8")[:-1].split(',')
                dte = self._date_converter(line[1])
                nt = self._nt_converter(line[2], line[3])
                yield dte, "{}_{}".format(k, line[0]), nt

class LMNDataReader3:
    
    def __init__(self, url):
        self.url = url
    
    def _date_converter(self, x):
        x = x.split()[0]
        if '-' in x:
            x = list(map(int, x.split('-')))
            x = datetime.date(x[0], x[1], x[2])
        else:
            logger.warn('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            x = float(x[2])/float(x[3])
        except:
            logger.warn('error trying to parse nt: {}'.format(x))
        
        return x
        
    def nt_data(self):
        """
        get data into a dict like
            data[source content][ticker] = [date, LMN words]
        """
            
        data = {}
        zf = zipfile.ZipFile(self.url)
        for k, fname in FILELIST3.items():
            data[k] = {}
            fcontent = zf.open(fname).readlines()
            for line in fcontent[1:]:
                line = line.decode("utf-8")[:-1].split(',')
                
            
                dte = self._date_converter(line[1])
                nt = self._nt_converter(line)
                
                yield dte, "{}_{}".format(k, line[0]), nt

                
class FFDataReader:
    
    def __init__(self, url):    
        self.url = url
    
    def _ff_date_converter(self, x):
        try: 
            x = datetime.date(int(x[:4]), int(x[4:6]), int(x[6:]))
        except Exception as E:
            logger.warnvv(repr(E))
        
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
                    logger.warn(repr(E))
                    continue
                    
                for n, ticker in enumerate(tickers):
                    yield dte, "{}_{}".format(k, ticker), float(line[n+1])                    

class CRSPDataFeeder:
    def __init__(self, url):
        self.url = url
    
    def _date_converter(self, x):
        return datetime.date(int(x[:4]), int(x[4:6]), int(x[6:]))
    
    def crsp_data(self):
        zf = zipfile.ZipFile(self.url)
        fcontent = zf.open('CRSP value-eighted index return.xlsx - WRDS.csv').readlines()
        for line in fcontent:   
            try:
                line = line[:-2].decode("utf-8").split(',')
                yield self._date_converter(line[0]), 'CRSP', float(line[1])
            except Exception as E:
                print(repr(E))
        
def lagger(x, lags, keep0=False):
    if keep0 is True:
        X = np.zeros((lags+1, len(x)))
        X[0] = x
        for i in range(1, lags+1):
            X[i][:-i] = x[i:]
        
    elif keep0 is False:
        X = np.zeros((lags, len(x)))
        for i in range(1, lags+1):
            X[i-1][:-i] = x[i:]
        
    return X   

def matrix_append(X, Y):
    """
    assumes 2 matrices NxM where Xm=Ym and you are appending over the N
    axis  
    """
    new = np.zeros((X.shape[0]+Y.shape[0], X.shape[1]))
    new[:X.shape[0]] = X
    new[X.shape[0]:] = Y
    return new

def zscore(x):
    x = (x-np.mean(x))/np.std(x)
    return x

class VARBundle:
    def __init__(self, ts, ticker, corpus):
        self.ticker = ticker
        self.corpus = corpus
        self.dts = ts.select('price_{}'.format(ticker))[0]
        self.price = ts.select('price_{}'.format(ticker))[1]
        self.crsp = ts.select('CRSP')[1][1:]
        self.returns = np.diff(np.log(self.price))
        self.adj_returns = zscore(self.returns-self.crsp)    
        self.nt = ts.select('{}_{}'.format(corpus, ticker))[1][1:]  
        self.sent = zscore(self.nt)
        self.friday = np.array([ts.select('friday')[1][1:]])
        self.jan = np.array([ts.select('january')[1][1:]])    
        self.NWD = np.array([ts.select('NWD')[1][1:]])
        
def make_var(ts, ticker, corpus):
    """In the JCF paper 
        - returns are ajusted of dividends and splits?
        - says CRSP is proxy for market but doesnt say if included 
        - 5 mentions (I dont do this) (pg 155)
        - negative tone is % of LMN words (pg 155)

        - coeficients multiplied by 100 when displayed 
        - added in the summ over the sent lags as this is included in 
            table 4 and the rolling regs (this is my best guess as to 
            how this was origionally implemented 
    """

    vb = VARBundle(ts, ticker, corpus)
   
    Xreturns = lagger(vb.returns, 5, keep0=False)
    Xsent = lagger(vb.sent, 5, keep0=False)
        
    sum_1to5 = np.array([np.sum(Xsent, 0)])    
    sum_2to5 = np.array([np.sum(Xsent[1:,:], 0)])
    
    logger.info('dims of Xsent, sum_1to5, sum_2to5 : {} {} {}'.format(
                    repr(Xsent.shape), repr(sum_1to5.shape), 
                    repr(sum_2to5.shape)))
    
    X = vb.jan
    for i in [vb.NWD, Xsent, sum_1to5, sum_2to5, Xreturns]:
        X = matrix_append(X, i)
    
    return X, vb.returns, vb
    
def pannal_var(ts):
    tickers = ts.tickers
    # deliberately excluding PFE 
    tickers = [i for i in tickers if i != 'PFE']
    for corpus in ['lexisnexis', 'web', 'web1', 'web2']:
        X, y, vb = make_var(ts, tickers[0], corpus)
        X = X.T
        for i in range(1, len(tickers)):
            X_, y_, vb = make_var(ts, tickers[i], corpus)
            X = matrix_append(X, X_.T)
            y = np.append(y, y_)
        
        res = sm.OLS(y, sm.tools.add_constant(X)).fit()
        logger.info(res.summary().as_text())
        logger.info(corpus)
        logger.info(repr(res.params))
        
def rolling_var(ts):
    """in the JCF paper
        - use a year rolling regression
        - the first lag is significant and negative 
        - the first lag is negative and the sum of lags 1-5 is significant 
            and negative 
        - the first lag is negative and the sum of lags 2-5 is significant 
            and positive
            
        TODO: 
            this needs splitting up
    """
    
    def _pt_periods(res):
        pt = { 'lag1': False, 'persistent': False, 'transient': False }
        if res.pvalues[3] < 0.05 and res.params[3] < 0:
            pt['lag1'] = True
            if res.pvalues[8] < 0.05 and res.params[8] < 0:
                pt['persistent'] = True
            if res.pvalues[9] < 0.05 and res.params[9] > 0:
                pt['transient'] = True
        return pt

    tickers = ts.tickers
    # deliberately exclude PFE
    tickers = [i for i in tickers if i != 'PFE']
    
    results = {}
    for corpus in ['lexisnexis', 'web1', 'web2', 'web']:
        results[corpus] = {}
        for ticker in tickers:            
            X, y, vb = make_var(ts, ticker, corpus)
            
            results[corpus][ticker] = { 
                'lag1_significant':  np.zeros(len(y)),
                'persistent': np.zeros(len(y)),
                'transient': np.zeros(len(y))
                }
            
            for i in range(125, X.shape[1]):
                X_ = X[:,i-125:i]
                y_ = y[i-125:i]
                res = sm.OLS(y_, X_.T).fit()
                
                pt = _pt_periods(res)
                if pt['lag1']:
                    results[corpus][ticker]['lag1_significant'][i-125:i] += 1
                if pt['persistent']:
                    results[corpus][ticker]['persistent'][i-125:i] += 1
                if pt['transient']:
                    results[corpus][ticker]['transient'][i-125:i] += 1
                        
    for corpus in results.keys():
        for ticker in results['corpus'].keys():
            fig = figure()
            axs = [fig.add_subplot(3,1,i) for i in range(1,4)]
            axs[0].plot(results[corpus][ticker]['lag1_significant'])
            axs[1].plot(results[corpus][ticker]['persistent'])
            axs[2].plot(results[corpus][ticker]['transient'])
            show()
            
def descriptive_statistics(ts):
    """
    replicates the descriptive stats tab of the spreadsheet
    """
    tickers = sorted(ts.tickers)
    for corpus in ['lexisnexis', 'web']:
        for ticker in tickers:
            vb = VARBundle(ts, ticker, corpus)
                
            print(ticker, corpus, np.mean(vb.nt), np.median(vb.nt),  np.max(vb.nt), 
                    len([i for i in vb.nt if i==0]), len(vb.nt))

FACTORY = {
    'pannal_var': pannal_var,
    'rolling_var': rolling_var,
    'descriptive_statistics': descriptive_statistics
    }
    
if __name__ == '__main__':  
    
    # init logging based on user params 
    function = FACTORY.get(sys.argv[1])
    if function is None:
        logger.error('cant find function : {}'.format(function))
        
    logger.info('running {}'.format(sys.argv[1]))
    
    #prep data
    min_date = datetime.date(2014, 1, 1)
    max_date = datetime.date(2015, 8, 23)
    url = 'data_for_financial_sentiment_paper.zip'
    
    ts = Tseries(date_range(min_date, max_date))
    
    #trading days first so we can exclude non trading days 
    ts.add(StockPriceDataFeeder(url).sp_data())
    ts.remove_null()
    
    ts.add(LMNDataReader3(url).nt_data())
    ts.add(FFDataReader(url).FF_data())
    ts.add(CRSPDataFeeder(url).crsp_data())
     
    ts.remove_null()
    ts.pad_missing()
    ts.dummy_vars(lambda x: x.weekday()==0, 'NWD')
    ts.dummy_vars(lambda x: x.weekday()==4, 'friday')
    ts.dummy_vars(lambda x: x.month==1, 'january')

    # do analysis
    function(ts)

    """


def get_crsp_data():
    with open('..
    /Downloads/CRSP value-eighted index return.xlsx - WRDS.csv', 'rb') as inp:
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








