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
	
def modelEvaluation(y_test,y_pred):

	from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
	
	confusionMatrix = confusion_matrix(y_test,y_pred)
	
	accuracyScore = accuracy_score(y_test,y_pred)
	
	print(classification_report(y_test,y_pred))
	
	return confusionMatrix, accuracyScore