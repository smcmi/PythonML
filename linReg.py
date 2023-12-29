def train_validation_test_split(X,y,train_size=70,validation_size=15,test_size=15,random_state=101):
	
	"""Splits features and labels into train, test, and validation sets. Tuple unpacking:
	   X_train, X_validation, X_test, y_train, y_validation, y_test = ...
	   train_validation_test_split(X,y,train_size,validation_size,test_size,random_state)
	   
	   sum([train_size, validation_size, test_size]) must be 1 (for proportions) or 100 (for percentages).
	"""
	size_total = sum([train_size,validation_size,test_size])
	
	if size_total not in [1,100]:
		print('The sum of the train, validation, and test sizes must be unity (percentages OK)')
		print('Existing inputs have a total of ' + str(size_total))
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
	
def scale(X_train,X_test):
	"""Scales train and test features with StandardScaler"""

	from sklearn.preprocessing import StandardScaler
	
	stdSclr = StandardScaler()
	X_train = stdSclr.fit_transform(X_train)
	X_test = stdSclr.transform(X_test)
	
	return X_train, X_test
	
def lassoRegression(X_train,X_test,y_train,degree,eps,n_alphas,cv):
	"""Performs LASSO regression with cross-validation on features 'X' and known labels 'y' with polynomial feature transformation of degree 'degree', where RSS plus L1 penalty (sum of abs. val. of coeff's) is minimized. Performs scaling on both train and test features. 
	"""
	
	from sklearn.linear_model import LassoCV
	
	return X_train, X_test, LassoCV(eps=eps, n_alphas=n_alphas, cv=cv).fit(X_train,y_train)
	
# def ridgeRegression(X_train,X_test,y_train,degree=3,
	
def runLasso(X,y,degree,train_size,random_state=101,eps=0.1,n_alphas=100,cv=5):
	
	tFactor = 1
	if train_size > 1:
		tFactor = 0.01
	
	test_size = 1-train_size*tFactor
		
	X = polyFeatures(X,degree=degree)
	
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
	
	X_train, X_test = scale(X_train,X_test)
	
	X_train, X_test, lasso_model = lassoRegression(X_train, X_test,y_train,degree=degree,eps=eps,n_alphas=n_alphas,cv=cv)
	
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	import numpy as np
	
	y_pred = lasso_model.predict(X_test)
	
	MAE = mean_absolute_error(y_test,y_pred)
	RSME = np.sqrt(mean_squared_error(y_test,y_pred))
	
	return lasso_model, MAE, RSME
	