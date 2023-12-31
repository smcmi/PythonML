def runGridSearch(X, y, degree, train_size, param_grid={'alpha':[0.1,1,5,10,50,100],'l1_ratio':[0.1,0.5,0.7,0.9,0.95,0.99,1]}, random_state=101, cv=5, scoring='neg_mean_squared_error', estimator='base_elastic_net_model', verbose=1, n_jobs=None):
	"""Performs polynomial regression of degree 'degree' on features 'X' with known labels 'y', including feature scaling and train/test split with percentage or proportion 'train_size' going into training set, and then minimizes objective function according to 'estimator' with parameter ranges given by the dictionary 'param_grid'."""
	
	from regressionSetup import poly_TTS_scale
	import pandas as pd
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X,y,degree,train_size,random_state)
	
	from sklearn.linear_model import GridSearchCV
	
	grid_model = GridSearchCV(estimator=estimator,param_grid=param_grid,scoring=scoring,cv=cv,verbose=verbose).fit(X_train,y_train)
	
	y_pred = grid_model.predict(X_test)
	
	cv_results = modelEvaluation(y_test,y_pred)
	
	return grid_model, cv_results