
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
        
    - I think the gt3 mentions makes a lot of difference for lexisnexis so we 
        need to reintroduce this for comparison 
"""

from pylab import figure, show, savefig
import numpy as np
import datetime
import statsmodels.api as sm
import zipfile
import logging
import sys 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__main__')

CORPRA = [ 'lexisnexis', 'web', 'web1', 'web2']

ROLLING_VAR_PARAMS = { 
    'JCF params': [250, 0.05, '../financial_sentiment_graphs/JCF_params_'],
    'med': [250, 0.05, '../financial_sentiment_graphs/6months_'],
    'short': [60, 0.05, '../financial_sentiment_graphs/3months_']
    }

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
            logger.warning('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            x = float(x)
        except:
            logger.warning('error trying to parse nt: {}'.format(x))
        
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
            logger.warning('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, T, x):
        
        try:
            #x = 100*(float(x)/float(T))
            x = float(x)
        except:
            logger.warning('error trying to parse nt: {}'.format(x))
        
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
            logger.warning('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            x = float(x[2])/float(x[3])
        except:
            logger.warning('error trying to parse nt: {}'.format(x))
        
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
            logger.warning(repr(E))
        
        return x
        
    def FF_data(self, greater_than=20140000):
        """
        get data into a dict like
            data[ff model] = [date, factors...]
            leave out the risk free rate as this is 0 throughout the study period 
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
                    logger.warning(repr(E))
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

class FFFactors:
    """Create a np matrix of ff factors where 
        self.factors -> NxM matrix where N is the number of factors and M is
            the length of the time series under study 
            factors are ordered by enumeration corresponding to the order they
            are provided by Fama and French on their website 
    """
    
    def __init__(self, model):
        series_names = sorted([i for i in ts.descriptives[2] if model in i])
        logger.info(series_names)
        dts, root_factor = ts.select(series_names[0])
        grouped_factors = np.array([root_factor])
        
        for name in series_names[1:]:
            factor = np.array([ts.select(name)[1]])
            grouped_factors = matrix_append(grouped_factors, factor)
        
        grouped_factors = grouped_factors[:,1:]
        dts = dts[1:]
        logger.info(grouped_factors.shape)
        
        self.dts = dts
        self.factors = grouped_factors
            
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
    
    logger.debug('dims of Xsent, sum_1to5, sum_2to5 : {} {} {}'.format(
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
    res_summary = []
    for corpus in CORPRA:
        X, y, vb = make_var(ts, tickers[0], corpus)
        X = X.T
        for i in range(1, len(tickers)):
            X_, y_, vb = make_var(ts, tickers[i], corpus)
            X = matrix_append(X, X_.T)
            y = np.append(y, y_)
        
        res = sm.OLS(y, sm.tools.add_constant(X)).fit()
        res_summary.append('{}\n{}\n'.format(corpus, res.summary().as_text()))
                        
        logger.info(res.summary().as_text())
        logger.info(corpus)
        
    with open('pannal_var_results.txt', 'w') as out:
        out.write('\n'.join(res_summary))
        
def rolling_var(ts):
    """in the JCF paper
        - use a year rolling regression
        - the first lag is significant and negative 
        - the first lag is negative and the sum of lags 1-5 is significant 
            and negative 
        - the first lag is negative and the sum of lags 2-5 is significant 
            and positive
        
        settings in the origional JCF paper 
        - days = 250
        - p = 0.05
        
        - this is quite function heavy so need to decide 
    """
 
    def _pt_periods(res):
        pt = { 'lag1': False, 'persistent': False, 'transient': False }
        if res.pvalues[3] < p and res.params[3] < 0:
            pt['lag1'] = True
            if res.pvalues[8] < p and res.params[8] < 0:
                pt['persistent'] = True
            if res.pvalues[9] < p and res.params[9] > 0:
                pt['transient'] = True
        return pt

    def _rv_iterator():
        tickers = ts.tickers
        # deliberately exclude PFE
        tickers = [i for i in tickers if i != 'PFE']
    
        for name, (days, p, path) in ROLLING_VAR_PARAMS.items():
            for corpus in CORPRA:
                for ticker in tickers:
                    yield name, days, p, path, corpus, ticker
        
    def _fit(X, y):
        results = {
            'lag1_significant':  np.zeros(len(y)),
            'persistent': np.zeros(len(y)),
            'transient': np.zeros(len(y))
            }            

        for i in range(days, X.shape[1]):
            X_ = X[:,i-days:i]
            y_ = y[i-days:i]
            res = sm.OLS(y_, X_.T).fit()
                
            pt = _pt_periods(res)   
            if pt['lag1']:
                results['lag1_significant'][i-days:i] += 1
            if pt['persistent']:
                results['persistent'][i-days:i] += 1
            if pt['transient']:
                results['transient'][i-days:i] += 1
                    
        return results

    def _count_periods(x):
        return sum([1 for i in x if i > 0])
        
    
    
    results = {}
    for name, days, p, path, corpus, ticker in _rv_iterator():
        id = '{}-{}-{}'.format(name, corpus, ticker)        
        X, y, vb = make_var(ts, ticker, corpus)
        results[id] = _fit(X, y)
    
    # now we have the results object terate over it and calculate
        # how many significant, persistent, and transient periods 
        # graph when they occur - excluding times when there are no significant
            # periods 
    
    totals = {}
    for id, series in results.items():
        experiment_id = '-'.join(id.split('-')[:2])

        if experiment_id not in totals:
            totals[experiment_id] = { 'lag1_significant': 0, 'persistent': 0, 
                                    'transient': 0 }
        
        if sum(series['lag1_significant']) == 0:
            logger.info('no significant periods for {}'.format(id))
            continue
        
        for k in totals[experiment_id]:
            totals[experiment_id][k] += _count_periods(series[k])
        
        '''
        fig = figure()
        axs = [fig.add_subplot(3,1,i) for i in range(1,4)]
        axs[0].plot(totals[experiment_id]['lag1_significant'])
        axs[1].plot(totals[experiment_id]['persistent'])
        axs[2].plot(totals[experiment_id]['transient'])
        savefig('{}{}'.format(path, name))        
        '''
    
    with open('rolling_var_results.txt', 'w') as out:
        for k, i in totals.items():
            logger.info('{} {}'.format(repr(k), repr(i)))
            out.write('{} {} {} {}\n'.format(k, i['lag1_significant'], 
                        i['persistent'], i['transient']
                        ))
    
def descriptive_statistics(ts):
    """
    replicates the descriptive stats tab of the spreadsheet
    """
    tickers = sorted(ts.tickers)
    for corpus in ['lexisnexis', 'web']:
        for ticker in tickers:
            vb = VARBundle(ts, ticker, corpus)
                
            print(ticker, corpus, np.mean(vb.nt), np.median(vb.nt),  
                    np.max(vb.nt), len([i for i in vb.nt if i==0]), 
                    len(vb.nt), vb.dts[0], vb.dts[-1])

def portfolio_analysis(ts):
    """I believe this is ust spliting the long/short portfolio based on the
        median sent and then regressing this against FF factor models 
        
        - need to understand how I worked this out the first time though 
        
        - adj returns or returns?
            - doenst state explicitly so have to assume adj_returns unless 
                there is some convention I am not aware of 
        - what is the exact trading strat?
            - long/short top half / bottom half portfolios. Hold for 10 days 
                or 250 days 
                
        TODO:
            need to refactor, this should be a class (so probably should the 
            other bits of analysis
    """
    
    def _get_series(ticker, corpus):
        vb = VARBundle(ts, ticker, corpus)
        sent = np.array([vb.sent])
        ret = np.array([vb.returns])                
        return sent, ret, vb
    
    def _get_port(corpus):
        tickers = ts.tickers
        # deliberately excluding PFE 
        tickers = [i for i in tickers if i != 'PFE']
        
        sent, ret, vb = _get_series(tickers[0], corpus)

        for ticker in tickers[1:]:
            sent_, ret_, vb = _get_series(ticker, corpus)
            sent = matrix_append(sent, sent_)
            ret = matrix_append(ret, ret_)
        
        sent = sent.T
        ret = ret.T
        Nstocks = len(tickers)
        T = len(sent)
        
        res = { 1: [], 10: [], 60: [], 120: [] }
        for t in range(1, T):
            order = np.argsort(sent[t-1])
            for k in res:
                if t+1+k < T:
                    holding_period_ret = np.mean(ret[t+1:t+1+k], 0)
                    lownt = np.mean(holding_period_ret[order][:10])
                    highnt = np.mean(holding_period_ret[order][10:])
                    res[k].append((lownt-highnt)/2)
        
        res = { k: np.array(i) for k, i in res.items() }
        return res             
    
    def _port_results_iter(port_returns):
        for corpus in port_returns:
            for holding_period in port_returns[corpus]:
                r = port_returns[corpus][holding_period]
                yield r, corpus, holding_period
    
    port_returns = { corpus: _get_port(corpus) for corpus in CORPRA }
    
    with open('portfolio_analysis_results.txt', 'w') as out:
        for r, corpus, holding_period in _port_results_iter(port_returns):
            out.write('{}\n'.format(','.join(list(map(str, 
                        [corpus, holding_period, np.mean(r), len(r)]
                        )))))
    
    with open('ff_regression_results.txt', 'w') as out:
        for model in ['3 factor', '5 factor']:
            ff = FFFactors(model)
            for r, corpus, holding_period in _port_results_iter(port_returns):
                logger.info('ff, y shapes = {} {}'.format(
                            repr(r.shape), repr(ff.factors.shape)
                            ))
                            
                X = ff.factors[:,:len(r)].T
                
                logger.info('X, y shapes = {} {}'.format(
                            repr(r.shape), repr(X.shape)
                            ))
                
                res = sm.OLS(r, sm.tools.add_constant(X)).fit()
                out.write('{} - {}\n{}\n\n'.format(corpus, holding_period,
                                res.summary().as_text()))
                                     
FACTORY = {
    'pannal_var': pannal_var,
    'rolling_var': rolling_var,
    'descriptive_statistics': descriptive_statistics,
    'portfolio_analysis': portfolio_analysis
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








