
import csv
from pylab import figure, show
import numpy as np
import datetime
#import statsmodels.api as sm
import math
import zipfile

FILELIST = {
    'web1': ['websites_AAPL_IBM_MSFT_IBM.csv','websites_everything_else.csv'],
    'web2': ['just_social_media_AAPL_IBM_MSFT_VZ.csv','just_social_media_everything_else.csv'],
    'lexisnexis': ['everything_else.csv - results-20160926-191305.csv.csv','AAPL_MSFT_VZ_AAPL.csv - results-20160926-190556.csv.csv'],
    'webAll': ['SOCIAL_AAPL_IBM_MSFT_VZ.csv - results-20160927-183311.csv.csv','SOCIALeverything_else.csv - results-20160927-183703.csv.csv']
    }

class LMNDataReader:

    def _date_converter(self, x):
        x = x.split()[0]
        
        if b'-' in x:
            x = list(map(int, x.split(b'-')))
            x = datetime.date(x[0], x[1], x[2])
        elif b'/' in x:
            x = list(map(int, x.split(b'/')))
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
    
    def __init__(self):
        data = {}
        zf = zipfile.ZipFile('data_for_financial_sentiment_paper.zip')
        for k, names in FILELIST.items():
            data[k] = {}
            for fname in names:
                fcontent = zf.open(fname).readlines()
                for line in fcontent:
                    line = line[:-1].split(b',')
                    dte = self._date_converter(line[1])
                    nt = self._nt_converter(line[2])
                    
                    if line[0] not in data[k]:
                        data[k][line[0]] = []
                        
                    data[k][line[0]].append([dte, nt])
    
    

        
                
                    
                
l = LMNDataReader()
            
"""
        
        with open('../Downloads/websites_AAPL_IBM_MSFT_IBM.csv', 'rb') as inp: # this is web 1
        #with open('../Downloads/just_social_media_AAPL_IBM_MSFT_VZ.csv', 'rb') as inp: # this is web 2
        #with open('../Downloads/everything_else.csv - results-20160926-191305.csv.csv', 'rb') as inp: # this is lexis nexis
        #with open('../Downloads/SOCIAL_AAPL_IBM_MSFT_VZ.csv - results-20160927-183311.csv.csv', 'rb') as inp:
            for i in csv.reader(inp):
                dte = i[1]
                stock = i[0]
                i[0] = dte
                i[1] = stock
                
                #try:
                #    i0 = i[0].split()[0]
                #    i0 = i0.split('/')
                #    i0 = map(int, i0)
                #    i[0] = datetime.date(i0[2], i0[0], i0[1])
                try:
                    i0 = i[0].split()[0]
                    i0 = i0.split('-')
                    i0 = map(int, i0)
                    i[0] = datetime.date(i0[0], i0[1], i0[2])
                
                except Exception as E:
                    print repr(E)
                    continue

                if i[0] not in data: data[i[0]] = {}
                data[i[0]][i[1]] = i[2]
                
        with open('../Downloads/websites_everything_else.csv', 'rb') as inp: # this is web 1
        #with open('../Downloads/just_social_media_everything_else.csv', 'rb') as inp: # this is web 2
        #with open('../Downloads/AAPL_MSFT_VZ_AAPL.csv - results-20160926-190556.csv.csv', 'rb') as inp: # this is lexis nexis
        #with open('../Downloads/SOCIALeverything_else.csv - results-20160927-183703.csv.csv', 'rb') as inp:

            for i in csv.reader(inp):
                dte = i[1]
                stock = i[0]
                i[0] = dte
                i[1] = stock
                
                #try:
                #    i0 = i[0].split()[0]
                #    i0 = i0.split('/')
                #    i0 = map(int, i0)
                #    i[0] = datetime.date(i0[2], i0[0], i0[1])
                try:
                    i0 = i[0].split()[0]
                    i0 = i0.split('-')
                    i0 = map(int, i0)
                    i[0] = datetime.date(i0[0], i0[1], i0[2])
                except Exception as E:
                    print repr(E)
                    continue

                if i[0] not in data: data[i[0]] = {}
                data[i[0]][i[1]] = i[2]

        stocks = {}
        for n, i in data.items():
            for stock in i.keys():
                if stock not in stocks: stocks[stock] = None
        stocks = stocks.keys()
        
        dts = sorted(data.keys())
        lmn = []
        for dte in dts:
            row = []
            for stock in stocks:
                if stock in data[dte]: row.append(data[dte][stock])
                else: row.append(0)
            lmn.append(row)

        print stocks

        return [stocks, dts, lmn]

def zs(x): 
    return (np.array(x)-np.mean(x))/np.std(x)

def pprintres(res):
    for n, i in enumerate(res.params):
        print n, i, res.pvalues[n]
    print res.rsquared_adj

def get_ff_factors(dts, which):
    if which == 0: s = 'F-F_Research_Data_Factors_daily_CSV/F-F_Research_Data_Factors_daily.CSV'
    else: s = 'F-F_Research_Data_5_Factors_2x3_daily_CSV/F-F_Research_Data_5_Factors_2x3_daily.CSV'
    
    dts2, data = [[],[]]
    with open('../Downloads/'+s, 'rb') as inp:
        for line in csv.reader(inp):
            try: dte = datetime.date(int(line[0][:4]), int(line[0][4:6]), int(line[0][6:]))
            except Exception as E:
                print repr(E)
                print line
                
                continue
                                     
            if dte in dts:
                dts2.append(dte)
                if which == 0: data.append(map(float, line[1:4]))
                else:
                    data.append(map(float, line[1:6]))
    return [dts2, np.array(data)/100]
                


        
    
def get_stock_price_data2():
    data = []
    with open('../Downloads/stock_data.csv', 'rb') as inp:
    #with open('../Downloads/stock_volumes.csv', 'rb') as inp:
    #with open('../Downloads/volatility_processed.csv', 'rb') as inp:
        for line in csv.reader(inp):
            data.append(line)


    tickers = data[0][1:]
    data = np.array(data[1:]).T
    dts = data[0][1:]
    dts_ = []
    for n, dte in enumerate(dts):
        try:
            dte = dte.split('/')
            dte = map(int, dte)
            dte = datetime.date(dte[2], dte[1], dte[0])
            dts_.append(dte)
        except Exception as E:
            print repr(E)

    for i in data: print i
    
    prices = np.array(data[1:], dtype=float)
    prices = prices.T
    return [tickers, dts_, prices]
    

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
    
def map_data_together(a, b, name):
    for n, dte in enumerate(b[0]):
        try: a[dte][name] = b[1][n]
        except Exception as E:
            print 'CANT FINE DATE', dte, repr(E)
    
    return a

def prep_for_analysis(data, tickers_price, tickers_lmn):
    ticker_mapping = []
    for t in tickers_price:
        ticker_mapping.append(tickers_lmn.index(t))

    for n, i in enumerate(ticker_mapping):
        print tickers_price[n], tickers_lmn[i]
        
    dts = sorted(data.keys())
    dts_ = []
    price, lmn, crsp = [[],[],[]]
    for dte in dts:
        if len(data[dte].keys()) != 3: continue
        prow = []
        lrow = []
        print len(data[dte]['r']), len(data[dte]['lmn']), data[dte]['lmn'], data[dte]['r']
        for n, i in enumerate(ticker_mapping):
            prow.append(data[dte]['r'][n])
            lrow.append(data[dte]['lmn'][i])

        price.append(prow)
        lmn.append(lrow)
        crsp.append(data[dte]['crsp'])
        dts_.append(dte)
    dts = dts_
        
    price = np.array(price, dtype=float)
    lmn = np.array(lmn, dtype=float)
    crsp = np.array(crsp, dtype=float)
    january = []
    friday = []
    NWK = []
    for dte in dts:
        if dte.month == 1: january.append(1)
        else: january.append(0)
        if dte.weekday() == 4: friday.append(1)
        else: friday.append(0)
        if dte.weekday() == 0: NWK.append(1)
        else: NWK.append(0)
        
    return [dts, price, lmn, crsp, january, friday, NWK]


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


def daterange(start, finish, d, tag):
    while start <= finish:
        if start in d: d[start][tag] = 1.
        else: d[start] = { tag: 1. }
        start = start+datetime.timedelta(1)
    return d


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








