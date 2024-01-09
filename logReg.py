def runLogReg(X, y, degree, train_size, count_plot=True, Cs=10, cv=5, random_state=101, max_iter=1000000, l1_ratios=[0.1,0.5,0.7,0.9,0.95,0.99,1],solver='saga', penalty='elasticnet', scoring='neg_mean_squared_error'):
	
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	from regressionSetup import poly_TTS_scale
	from sklearn.linear_model import LogisticRegressionCV
	
	posRatio = y.sum()/len(y)
	prStr = str(int(np.round(posRatio*100,0))) + '% Class 1'
	if posRatio > 0.6 or posRatio < 0.4:
		print('Unbalanced data set: ' + prStr)
	else:
		print('Balanced data set: ' + prStr)
		
	if count_plot:
		sns.countplot(x=y)
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X=X,y=y,degree=degree,train_size=train_size,random_state=random_state)
	
	log_model = LogisticRegressionCV(Cs=Cs,cv=cv,random_state=random_state,max_iter=max_iter,l1_ratios=l1_ratios,solver=solver,penalty=penalty,scoring=scoring).fit(X_train,y_train)
	
	if train_size in [1,100]:
		y_pred = log_model.predict(X_train)
		y_test = y_train
	else:
		y_pred = log_model.predict(X_test)
		
	[confusionMatrix,accuracyScore] = modelEvaluation(y_test,y_pred)
	
	return log_model, confusionMatrix, accuracyScore
	
def runKNeighbors(X, y, train_size, degree=1, k_neighbors=list(range(1,20)), showResPlot = True, cv=5, random_state=101):

	from regressionSetup import poly_TTS_scale
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.model_selection import GridSearchCV
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X=X,y=y,degree=degree,train_size=train_size,random_state=random_state)
	
	param_grid = {'n_neighbors':k_neighbors}
	
	grid_model = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid,cv=cv,scoring='accuracy').fit(X_train,y_train)
	
	confusionMatrix, accuracyScore= modelEvaluation(y_test,grid_model.predict(X_test))
	
	if len(k_neighbors)>1 and showResPlot:
		import matplotlib.pyplot as plt
		plt.plot(k_neighbors,grid_model.cv_results_['mean_test_score'],'k.')
	
	return grid_model, confusionMatrix, accuracyScore
	
def runSVMClassifier(X, y, train_size, degree=1, param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}, random_state=101):
	
	from sklearn.svm import SVC
	from sklearn.model_selection import GridSearchCV
	from regressionSetup import poly_TTS_scale
	
	X_train, X_test, y_train, y_test = poly_TTS_scale(X=X,y=y,degree=degree,train_size=train_size,random_state=random_state)
	
	SVM_model = GridSearchCV(SVC(),param_grid).fit(X_train,y_train)
	# SVC parameters: kernel:['linear','poly' (+'degree'), 'rbf', 'sigmoid', 'precomputed'], C (regularization strength proportional to 1/C, L2 penalty), gamma: (kernel coefficient for 'rbf', 'poly', or 'sigmoid')
	
	return SVM_model	
	
def modelEvaluation(y_test,y_pred):

	from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
	
	confusionMatrix = confusion_matrix(y_test,y_pred)
	
	accuracyScore = accuracy_score(y_test,y_pred)
	
	print(classification_report(y_test,y_pred))
	
	return confusionMatrix, accuracyScore