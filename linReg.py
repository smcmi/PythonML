import importlib
import regressionSetup
importlib.reload(regressionSetup)

def runLasso(X, y, degree, train_size, random_state=101, eps=0.1, n_alphas=100, cv=5, max_iter=1000000, verbose=1):
	"""Performs polynomial regression of degree 'degree' on features 'X' with known labels 'y', including feature scaling and train/test split with percentage or proportion 'train_size' going into training set, and then minimizes RSS + L1 penalty (LASSO method, L1 is sum of abs. values of coeff's). Adjustable parameters are 'eps' [0.1] (ratio of alpha parameters), 'n_alphas' [100] (number of alpha parameters to test along regularization path), 'cv' [5] ("k" in k-fold cross-validation), and 'max_iter' [1e6] (maximum number of iterations in optimization scheme)."""
	
	print(X.shape)
	X_train, X_test, y_train, y_test = regressionSetup.poly_TTS_scale(X,y,degree,train_size,random_state)	
	print(X_train.shape)
	from sklearn.linear_model import LassoCV
	
	lasso_model = LassoCV(eps=eps, n_alphas=n_alphas, cv=cv, max_iter=max_iter).fit(X_train,y_train)
	
	if train_size in [1,100]:
		y_pred = lasso_model.predict(X_train)
		y_test = y_train
	else:
		y_pred = lasso_model.predict(X_test)
	
	MAE, RSME = modelEvaluation(y_test,y_pred)
	
	return lasso_model, MAE, RSME
	
def runElasticNet(X, y, degree, train_size, random_state=101, l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99,1], eps=0.001, n_alphas=100, cv=5, max_iter=1000000, tol=0.01, verbose=1, n_jobs=None):
	"""Performs polynomial regression of degree 'degree' on features 'X' with known labels 'y', including feature scaling and train/test split with percentage or proportion 'train_size' going into training set, and then minimizes RSS + alpha*L1 + (1-alpha)*L2 penalty. Adjustable parameters are 'eps' [0.1] (ratio of alpha parameters), 'n_alphas' [100] (number of alpha parameters to test along regularization path), 'cv' [5] ("k" in k-fold cross-validation), and 'max_iter' [1e6] (maximum number of iterations in optimization scheme)."""
	
	X_train, X_test, y_train, y_test = regressionSetup.poly_TTS_scale(X,y,degree,train_size,random_state)
	
	from sklearn.linear_model import ElasticNetCV
	
	elastic_model = ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, cv=cv, max_iter=max_iter, tol=tol, verbose=verbose).fit(X_train,y_train)
	
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	
	if train_size in [1,100]:
		y_pred = elastic_model.predict(X_train)
		y_test = y_train
	else:
		y_pred = elastic_model.predict(X_test)
	
	MAE, RSME = modelEvaluation(y_test,y_pred)
	
	return elastic_model, MAE, RSME
	
def modelEvaluation(y_test,y_pred):
	"""Evaluates model with appropriate metrics for the method used."""

	import inspect
	
	callerFunction = inspect.stack()[1][3]
	
	if callerFunction in ['runLasso' , 'runElasticNet']:
		from sklearn.metrics import mean_absolute_error, mean_squared_error
		import numpy as np
	
		MAE = mean_absolute_error(y_test,y_pred)
		RSME = np.sqrt(mean_squared_error(y_test,y_pred))
			
		return MAE, RSME
			
	if callerFunction == 'runGridSearch':
		import pandas as pd
		return pd.DataFrame(grid_model.cv_results_)
	
	
	