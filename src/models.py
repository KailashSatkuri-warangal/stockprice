from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def get_model(model_name="linear"):
	if model_name == "linear":
		return LinearRegression()
	elif model_name == "rf":
		return RandomForestRegressor(n_estimators=100, random_state=42)
	else:
		raise ValueError("Model must be 'linear' or 'rf'")
