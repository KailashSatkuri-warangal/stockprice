import pandas as pd

def preprocess(df):
	df = df.copy()
	df['Date'] = pd.to_datetime(df['Date'])
	df.sort_values('Date', inplace=True)
	df.reset_index(drop=True, inplace=True)
	return df
