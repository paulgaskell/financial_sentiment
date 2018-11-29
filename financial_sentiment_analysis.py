
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
        
        
    TODO:
        - need to get filtered reults from BigQuery - but we are on the way to 
            doing that I think 
        - how can we test the hypothesis that the issue is noise?
"""

from pylab import figure, show, savefig
import numpy as np
import datetime
import statsmodels.api as sm
import zipfile
import logging
import sys 

from data import (Tseries, LMNDataReader4, StockPriceDataFeeder, 
                    FFDataReader, CRSPDataFeeder)
from utils import date_range, zscore, matrix_append
from config import CORPRA, ROLLING_VAR_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__main__')
        
class VARBundle:
    
    def __init__(self, ts, ticker, corpus, mentions):
        self.ticker = ticker
        self.corpus = corpus
        self.dts = ts.select('price_{}'.format(ticker))[0]
        self.price = ts.select('price_{}'.format(ticker))[1]
        self.crsp = ts.select('CRSP')[1][1:]
        self.returns = np.diff(np.log(self.price))
        self.adj_returns = zscore(self.returns-self.crsp)    
        self.nt = ts.select('{}_{}_{}'.format(corpus, ticker, mentions))[1][1:]  
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

class VAR:
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

    @staticmethod
    def _lagger(x, lags, keep0=False):
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

    @staticmethod
    def get(ts, ticker, corpus):
        vb = VARBundle(ts, ticker, corpus, mentions)
       
        Xreturns = VAR._lagger(vb.returns, 5, keep0=False)
        Xsent = VAR._lagger(vb.sent, 5, keep0=False)
            
        sum_1to5 = np.array([np.sum(Xsent, 0)])    
        sum_2to5 = np.array([np.sum(Xsent[1:,:], 0)])
        
        logger.debug('dims of Xsent, sum_1to5, sum_2to5 : {} {} {}'.format(
                        repr(Xsent.shape), repr(sum_1to5.shape), 
                        repr(sum_2to5.shape)))
        
        X = vb.jan
        for i in [vb.NWD, Xsent, sum_1to5, sum_2to5, Xreturns]:
            X = matrix_append(X, i)
        
        return X, vb.returns, vb

class PanelVAR:   
    """Create panel regression with optional filtering
        the idea behind the filtering is that are different potential noises 
            - typical gaussian noise becasue the metrc is poor
            - catestrophic faliure because we are counting bad messages
        
    """
    
    @staticmethod
    def get(ts):
        tickers = ts.tickers
        # deliberately excluding PFE 
        tickers = [i for i in tickers if i != 'PFE']
        res_summary, filtered_res_summary = [], []
        for corpus in CORPRA:
            X, y, vb = VAR.get(ts, tickers[0], corpus)
            X = X.T
            for i in range(1, len(tickers)):
                X_, y_, vb = VAR.get(ts, tickers[i], corpus)
                X = matrix_append(X, X_.T)
                y = np.append(y, y_)
            
            filtered_X, filtered_y = [], []
            for t in range(len(y)):
                if .2 < abs(X[t][2]) < 2:
                    filtered_X.append(X[t])
                    filtered_y.append(y[t])
            
            res = sm.OLS(y, sm.tools.add_constant(X)).fit()
            res_summary.append('{}\n{}\n'.format(corpus, 
                                    res.summary().as_text()))
            filtered_res = sm.OLS(filtered_y, sm.tools.add_constant(
                                    filtered_X)).fit()
            filtered_res_summary.append('{}\n{}\n'.format(
                                    corpus, filtered_res.summary().as_text()))
            
            logger.info(corpus)
            logger.info(res.summary().as_text())
            logger.info(filtered_res.summary().as_text())
            
        with open('panel_var_results.txt', 'w') as out:
            out.write('\n'.join(res_summary))
            out.write('\n'.join(filtered_res_summary))
             
             
class RollingVAR:
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

    @staticmethod
    def _pt_periods(res, p):
        pt = { 'lag1': False, 'persistent': False, 'transient': False }
        if res.pvalues[3] < p and res.params[3] < 0:
            pt['lag1'] = True
            if res.pvalues[8] < p and res.params[8] < 0:
                pt['persistent'] = True
            if res.pvalues[9] < p and res.params[9] > 0:
                pt['transient'] = True
        return pt

    @staticmethod
    def _rv_iterator(ts):
        tickers = ts.tickers
        # deliberately exclude PFE
        tickers = [i for i in tickers if i != 'PFE']
    
        for name, (days, p, path) in ROLLING_VAR_PARAMS.items():
            for corpus in CORPRA:
                for ticker in tickers:
                    yield name, days, p, path, corpus, ticker
    
    @staticmethod
    def _fit(X, y, days, p):
        results = {
            'lag1_significant':  np.zeros(len(y)),
            'persistent': np.zeros(len(y)),
            'transient': np.zeros(len(y)),
            }            

        for i in range(days, X.shape[1]):
            X_ = X[:,i-days:i]
            y_ = y[i-days:i]
            res = sm.OLS(y_, X_.T).fit()
                
            pt = RollingVAR._pt_periods(res, p)   
            if pt['lag1']:
                results['lag1_significant'][i-days:i] += 1
            if pt['persistent']:
                results['persistent'][i-days:i] += 1
            if pt['transient']:
                results['transient'][i-days:i] += 1
        
        return results
        
    @staticmethod
    def _count_periods(x):
        return sum([1 for i in x if i > 0])
        
    
    @staticmethod
    def get(ts):
        results = {}
        for name, days, p, path, corpus, ticker in RollingVAR._rv_iterator(ts):
            id = '{}-{}-{}'.format(name, corpus, ticker)        
            X, y, vb = VAR.get(ts, ticker, corpus)
            results[id] = RollingVAR._fit(X, y, days, p)
        
        # now we have the results object terate over it and calculate
            # how many significant, persistent, and transient periods 
            # graph when they occur - excluding times when there are no significant
                # periods 
        
        totals = {}
        for id, series in results.items():
            experiment_id = '-'.join(id.split('-')[:2])

            if experiment_id not in totals:
                totals[experiment_id] = { 'lag1_significant': 0, 'persistent': 0, 
                                            'transient': 0, 'days': 0 }
            
            for k in totals[experiment_id]:
                if k == 'days':
                    totals[experiment_id][k] += len(series['lag1_significant'])
                else:
                    totals[experiment_id][k] += RollingVAR._count_periods(
                                                    series[k])
            """
            if sum(series['lag1_significant']) == 0:
                logger.info('no significant periods for {}'.format(id))
                continue
            
            fig = figure()
            axs = [fig.add_subplot(3,1,i) for i in range(1,4)]
            axs[0].plot(totals[experiment_id]['lag1_significant'])
            axs[1].plot(totals[experiment_id]['persistent'])
            axs[2].plot(totals[experiment_id]['transient'])
            savefig('{}{}'.format(path, name))        
            """
        
        with open('rolling_var_results.txt', 'w') as out:
            for k, i in totals.items():
                logger.info('{} {}'.format(repr(k), repr(i)))
                total_days = float(i['days'])
                
                out.write('{}\n'.format(','.join(list(map(str, [k, 
                            i['lag1_significant'], i['persistent'],
                            i['transient'], total_days, 
                            i['lag1_significant']/total_days, 
                            i['persistent']/total_days, 
                            i['transient']/total_days
                            ])))))

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

class PortfolioAnalysis:
    """I believe this is ust spliting the long/short portfolio based on the
        median sent and then regressing this against FF factor models 
        
        - need to understand how I worked this out the first time though 
        
        - adj returns or returns?
            - doenst state explicitly so have to assume adj_returns unless 
                there is some convention I am not aware of 
        - what is the exact trading strat?
            - long/short top half / bottom half portfolios. Hold for 10 days 
                or 250 days 
                
    """
    
    @staticmethod
    def _get_series(ticker, corpus):
        vb = VARBundle(ts, ticker, corpus)
        sent = np.array([vb.sent])
        ret = np.array([vb.returns])                
        return sent, ret, vb
    
    @staticmethod
    def _get_port(corpus):
        tickers = ts.tickers
        # deliberately excluding PFE 
        tickers = [i for i in tickers if i != 'PFE']
        
        sent, ret, vb = PortfolioAnalysis._get_series(tickers[0], corpus)

        for ticker in tickers[1:]:
            sent_, ret_, vb = PortfolioAnalysis._get_series(ticker, corpus)
            sent = matrix_append(sent, sent_)
            ret = matrix_append(ret, ret_)
        
        sent = sent.T
        ret = ret.T
        Nstocks = len(tickers)
        T = len(sent)
        
        res = { 1: [], 10: [], 60: [], 120: [], 250: [] }
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
    
    @staticmethod
    def _port_results_iter(port_returns):
        for corpus in port_returns:
            for holding_period in port_returns[corpus]:
                r = port_returns[corpus][holding_period]
                yield r, corpus, holding_period

    @staticmethod
    def get(ts):
        port_returns = { corpus: PortfolioAnalysis._get_port(corpus) 
                            for corpus in CORPRA }
        
        with open('portfolio_analysis_results.txt', 'w') as out:
            iter = PortfolioAnalysis._port_results_iter(port_returns)
            for r, corpus, holding_period in iter:
                out.write('{}\n'.format(','.join(list(map(str, 
                            [corpus, holding_period, np.mean(r), len(r)]
                            )))))
        
        with open('ff_regression_results.txt', 'w') as out:
            for model in ['3 factor', '5 factor']:
                ff = FFFactors(model)
                iter = PortfolioAnalysis._port_results_iter(port_returns)
                for r, corpus, holding_period in iter:
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
    'panel_var': PanelVAR.get,
    'rolling_var': RollingVAR.get,
    'descriptive_statistics': descriptive_statistics,
    'portfolio_analysis': PortfolioAnalysis.get
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
    
    ts.add(LMNDataReader4(url).nt_data())
    ts.add(FFDataReader(url).FF_data())
    ts.add(CRSPDataFeeder(url).crsp_data())
     
    ts.remove_null()
    ts.pad_missing()
    ts.dummy_vars(lambda x: x.weekday()==0, 'NWD')
    ts.dummy_vars(lambda x: x.weekday()==4, 'friday')
    ts.dummy_vars(lambda x: x.month==1, 'january')
    
    print(ts.descriptives)
    
    # do analysis
    function(ts)








