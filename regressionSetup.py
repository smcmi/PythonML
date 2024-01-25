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
	print('pttss')
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
