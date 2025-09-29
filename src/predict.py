import pandas as pd
from .features import create_features
from .preprocess import preprocess

def predict_next(model, df):
	df = preprocess(df)
	df = create_features(df)
	latest = df[['MA5','MA10','Volatility','Volume']].iloc[-1].values.reshape(1,-1)
	return model.predict(latest)[0]
