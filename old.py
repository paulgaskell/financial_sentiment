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