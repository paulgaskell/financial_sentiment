
"""
Functionlity for loading in different datasets to a consistent Tseris object 
"""


import zipfile
import logging
import datetime
import logging 
import numpy as np

from config import (FILELIST, FILELIST2, FILELIST3, FFLIST, STOCKPRICELIST,
                    FILELIST4, FILTERS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__main__')

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

class LMNDataReader4:
        
    def __init__(self, url):
        self.url = url
        self.names = FILTERS
    
    def _date_converter(self, x):
        if '-' in x:
            x = x.split()[0]
            x = list(map(int, x.split('-')))
            x = datetime.date(x[0], x[1], x[2])
        else:
            logger.warning('error trying to parse date: {}'.format(x))
        
        return x
    
    def _nt_converter(self, x):
        try:
            nt = np.array(list(map(float, x[2:7])))
            total = np.array(list(map(float, x[7:])))
            x = nt/total
            x = np.nan_to_num(x)
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
        for k, fname in FILELIST4.items():
            data[k] = {}
            fcontent = zf.open(fname).readlines()
            for line in fcontent[1:]:
                line = line.decode("utf-8")[:-1].split(',')
                
                dte = self._date_converter(line[1])
                nts = self._nt_converter(line)
                
                for n, nt in enumerate(nts):
                    yield dte, "{}_{}_{}".format(k, line[0], self.names[n]), nt
                