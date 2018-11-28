CORPRA = [ 'lexisnexis', 'web', 'web1', 'web2']

FILTERS = [ 'gt0', 'gt1', 'gt2', 'gt3', 'gt5' ]

ROLLING_VAR_PARAMS = { 
    'JCF params': [250, 0.05, '../financial_sentiment_graphs/JCF_params_'],
    'med': [120, 0.05, '../financial_sentiment_graphs/6months_'],
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

FILELIST4 = {
    'lexisnexis': 'word_counts_lexisnexis_v2',
    'web': 'word_counts_web_v2',
    'web1': 'word_counts_web1_v2',
    'web2': 'word_counts_web2_v2'
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