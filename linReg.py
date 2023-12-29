def train_validation_test_split(X,y,train_size=70,validation_size=15,test_size=15,random_state=101):
	
	"""Splits features and labels into train, test, and validation sets. Tuple unpacking:
	   X_train, X_validation, X_test, y_train, y_validation, y_test = ...
	   train_validation_test_split(X,y,train_size,validation_size,test_size,random_state)
	   
	   sum([train_size, validation_size, test_size]) must be 1 (for proportions) or 100 (for percentages).
	"""
	
	import numpy as np
	
	size_total = sum([train_size,validation_size,test_size])
	
	if size_total not in [1,100]:
		print('The sum of the train, validation, and test sizes must be unity (percentages are OK).')
		pStr = ""
		if any(np.array([train_size,validation_size,test_size]) > 1):
			pStr = "%"
		print('Existing inputs have a total of ' + str(np.round(size_total,3)) + pStr)
		return
	
	if train_size > 1:
		train_size /= 100
		
	if validation_size > 1:
		validation_size /= 100
		
	if test_size > 1:
		test_size /= 100 
	
	from sklearn.model_selection import train_test_split
	
	X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=1-train_size, random_state=random_state)
	X_validation, X_test, y_validation, y_test = train_test_split(X_vt, y_vt, test_size=test_size/(test_size+validation_size), random_state=random_state)
	
	return X_train, X_validation, X_test, y_train, y_validation, y_test
	
def polyFeatures(X,degree=3):
	"""Converts features to polynomial of degree 'degree'"""
	
	from sklearn.preprocessing import PolynomialFeatures
	
	pf = PolynomialFeatures(degree=degree,include_bias=False)
	X = pf.fit_transform(X)
	
	return X
	
def scaling(X_train,X_test):
	"""Scales train and test features with StandardScaler"""

	from sklearn.preprocessing import StandardScaler
	import numpy as np
	
	stdSclr = StandardScaler()
	X_train = stdSclr.fit_transform(X_train)
	
	if X_test.any().any():							# Test set exists
		X_test = stdSclr.transform(X_test)
	else:											# No test set
		X_test = np.array([0])
		
	return X_train, X_test
	
def poly_TTS_scale(X,y,degree,train_size,random_state):
	"""Performs polynomial of degree 'degree' transformation on features 'X', train_test_split with train proportion/percentage 'train_size', and scales features by training set"""
	
	import numpy as np
	
	X_train = polyFeatures(X,degree=degree)
	
	if train_size in [1,100]:						# Will train on full model, no test set. Scale and return.
		
		X_train = X
		y_train = y
		X_test = np.zeros(X.shape)
		y_test = np.zeros(y.shape)
		
	else:											# Perform train_test_split to have test set, scale, and return.

		tFactor = 1
		if train_size > 1:
			tFactor = 0.01
		
		test_size = 1-train_size*tFactor
				
		from sklearn.model_selection import train_test_split
		
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
		
	X_train, X_test = scaling(X_train,X_test)
	
	return X_train, X_test, y_train, y_test
	
def runLasso(X, y, degree, train_size, random_state=101, eps=0.1, n_alphas=100, cv=5, max_iter=1000000, verbose=1):
	"""Performs polynomial regression of degree 'degree' on features 'X' with known labels 'y', including feature scaling and train/test split with percentage or proportion 'train_size' going into training set, and then minimizes RSS + L1 penalty (LASSO method, L1 is sum of abs. values of coeff's). Adjustable parameters are 'eps' [0.1] (ratio of alpha parameters), 'n_alphas' [100] (number of alpha parameters to test along regularization path), 'cv' [5] ("k" in k-fold cross-validation), and 'max_iter' [1e6] (maximum number of iterations in optimization scheme)."""
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X,y,degree,train_size,random_state)	
	
	from sklearn.linear_model import LassoCV
	
	lasso_model = LassoCV(eps=eps, n_alphas=n_alphas, cv=cv, max_iter=max_iter).fit(X_train,y_train)
	
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	import numpy as np
	
	if train_size in [1,100]:
		y_pred = lasso_model.predict(X_train)
		y_test = y_train
	else:
		y_pred = lasso_model.predict(X_test)
	
	MAE = mean_absolute_error(y_test,y_pred)
	RSME = np.sqrt(mean_squared_error(y_test,y_pred))
	
	return lasso_model, MAE, RSME
	
def runElasticNet(X, y, degree, train_size, random_state=101, l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99,1], eps=0.001, n_alphas=100, cv=5, max_iter=1000000, tol=0.01, verbose=1):
	"""Performs polynomial regression of degree 'degree' on features 'X' with known labels 'y', including feature scaling and train/test split with percentage or proportion 'train_size' going into training set, and then minimizes RSS + alpha*L1 + (1-alpha)*L2 penalty. Adjustable parameters are 'eps' [0.1] (ratio of alpha parameters), 'n_alphas' [100] (number of alpha parameters to test along regularization path), 'cv' [5] ("k" in k-fold cross-validation), and 'max_iter' [1e6] (maximum number of iterations in optimization scheme)."""
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X,y,degree,train_size,random_state)
	
	from sklearn.linear_model import ElasticNetCV
	
	elastic_model = ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, cv=cv, max_iter=max_iter, tol=tol, verbose=verbose).fit(X_train,y_train)
	
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	import numpy as np
	
	if train_size in [1,100]:
		y_pred = elastic_model.predict(X_train)
		y_test = y_train
	else:
		y_pred = elastic_model.predict(X_test)
	
	MAE = mean_absolute_error(y_test,y_pred)
	RSME = np.sqrt(mean_squared_error(y_test,y_pred))
	
	return elastic_model, MAE, RSME