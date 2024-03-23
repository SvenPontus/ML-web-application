from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
                             

# My workflow is to have every models like methods

# Machine learning Regression
class MLR():
    """For Continous target value, scale, gridsearch for best hyper parameters.
    Opportunity to get cost function report and dump best model"""
    def __init__(self,y,X):
        self.y = y
        self.X = X
        (self.X_train, 
         self.X_test, 
         self.y_train, 
         self.y_test) = train_test_split(self.X, 
                                         self.y, 
                                         test_size=0.33, 
                                         random_state=101)
        self.scaler = StandardScaler()
        self.y_pred = None
        self.best_model_params = None
        self.model_r2_score = None
        self.model = None
        self.name = None

    # Linear Regression
    def linear_regression(self):
        """LINEAR REGRESSION, 0 hyper parameters"""
        from sklearn.linear_model import LinearRegression
        # Name of the model
        self.name = "linear_regression"
        # call Linearregression class
        LiR_model = LinearRegression()
        operations = [('scaler',self.scaler),('linear_regression',LiR_model)] 
        scaled_LiR_model_pipe = Pipeline(operations)
        param_grid = {} 
        # Gridsearch for find best hyperparameters
        LiR_best_model = GridSearchCV(scaled_LiR_model_pipe,
                                      param_grid=param_grid,
                                      cv=10,
                                      scoring="neg_mean_absolute_error")
        # train model
        LiR_best_model.fit(self.X_train,self.y_train)
        # send best model to constructor
        self.model = LiR_best_model
        # send best params to constructor
        self.best_model_params = LiR_best_model.best_params_
        # Send y_pred to constructor
        self.y_pred = LiR_best_model.predict(self.X_test)
                    
    # LASSO
    def lasso_method(self):
        """LASSO, param grid: lasso__alpha': [0.001, 0.01, 0.1, 1, 10]"""
        from sklearn.linear_model import Lasso
        self.name = "lasso"
        lasso_model = Lasso()
        operations = [('scaler',self.scaler),('lasso',lasso_model)] 
        scaled_lasso_model_pipe = Pipeline(operations)
        param_grid = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10]} 
        lasso_best_model = GridSearchCV(scaled_lasso_model_pipe,
                                      param_grid=param_grid,
                                      cv=10,
                                      scoring="neg_mean_absolute_error")
        
        lasso_best_model.fit(self.X_train,self.y_train)
        self.model = lasso_best_model
        self.best_model_params = lasso_best_model.best_params_
        self.y_pred = lasso_best_model.predict(self.X_test)


    # RIDGE
    def ridge_method(self):
        """RIDGE, param grid: ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]"""
        from sklearn.linear_model import Ridge
        self.name = "ridge"
        ridge_model = Ridge()
        operations = [('scaler',self.scaler),('ridge',ridge_model)] 
        scaled_ridge_model_pipe = Pipeline(operations)
        param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
        ridge_best_model = GridSearchCV(scaled_ridge_model_pipe,
                                      param_grid=param_grid,
                                      cv=10,
                                      scoring="neg_mean_absolute_error")
        
        ridge_best_model.fit(self.X_train,self.y_train)
        self.model = ridge_best_model
        self.best_model_params = ridge_best_model.best_params_
        self.y_pred = ridge_best_model.predict(self.X_test)

    # ELASTIC NET
    def elasticnet_method(self):
        """ELASTICNET, param grid: elasticnet__l1_ratio: [.1, .5, .7, .9, .95, .99, 1],
            'elasticnet__alpha': [0.01, 0.1, 1, 10, 100],  
            'elasticnet__max_iter': [10_000]"""
        from sklearn.linear_model import ElasticNet
        self.name = "elasticnet"
        elasticnet_model = ElasticNet()
        operations = [('scaler',self.scaler),('elasticnet',elasticnet_model)] 
        scaled_elasticnet_model_pipe = Pipeline(operations)
        param_grid = {
            'elasticnet__l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'elasticnet__alpha': [0.01, 0.1, 1, 10, 100],  
            'elasticnet__max_iter': [10_000]
        }
        elasticnet_best_model = GridSearchCV(scaled_elasticnet_model_pipe,
                                param_grid=param_grid,
                                cv=10,
                                scoring="neg_mean_absolute_error")
        
        elasticnet_best_model.fit(self.X_train,self.y_train)
        self.model = elasticnet_best_model
        self.best_model_params = elasticnet_best_model.best_params_
        self.y_pred = elasticnet_best_model.predict(self.X_test)

    # SVR
    def svr_method(self):
        """SVR, param grid: 'svr__C': np.logspace(0, 1, 10),  
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'svr__degree': np.arange(1, 9),  
            'svr__gamma': ['scale', 'auto'] """
        from sklearn.svm import SVR
        self.name = "svr"
        svr_model = SVR()
        operations = [('scaler',self.scaler),('svr',svr_model)]
        scaled_svr_model_pipe = Pipeline(operations)
        param_grid = {
            'svr__C': np.logspace(0, 1, 10),  
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'svr__degree': np.arange(1, 9),  
            'svr__gamma': ['scale', 'auto'] 
        }
        svr_model_best_model = GridSearchCV(scaled_svr_model_pipe,
                                param_grid=param_grid,
                                cv=10,
                                scoring="neg_mean_absolute_error")
        
        svr_model_best_model.fit(self.X_train,self.y_train)
        self.model = svr_model_best_model
        self.best_model_params = svr_model_best_model.best_params_
        self.y_pred = svr_model_best_model.predict(self.X_test)
        
    # REPORT ALL MODELS
    def report_cost_func(self):
        """MAE, MSE, RMSE, r2 score"""
        MAE = mean_absolute_error(y_true=self.y_test, y_pred=self.y_pred)
        MSE = mean_squared_error(y_true=self.y_test, y_pred=self.y_pred)
        RMSE = np.sqrt(MSE)
        # send r2 score to constructor
        self.model_r2_score = r2_score(y_true=self.y_test, y_pred=self.y_pred)
        return f"MAE : {MAE}\n MSE : {MSE}\nRMSE : {RMSE}\nModel_r2_score : "\
               f"{self.model_r2_score}\nBest params : {self.best_model_params}"\
                f"\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"

    def send_model_r2_score(self):
        return self.model_r2_score
    
    # dump model method
    def dump_best_model(self):
        from joblib import dump
        final_model = self.model.fit(self.X,self.y)
        dump(final_model, f"final_model_{self.name}.joblib")
        

# Machine learning classification
class MLC():
    """For Categorical target value, scale, gridsearch for best hyper parameters.
    Opportunity to get cost function report and dump best model"""
    def __init__(self,y,X):
        self.y = y
        self.X = X
        (self.X_train, 
         self.X_test, 
         self.y_train, 
         self.y_test) = train_test_split(self.X, 
                                         self.y, 
                                         test_size=0.33, 
                                         random_state=101)
        self.scaler = StandardScaler()
        self.y_pred = None
        self.best_model_params = None
        self.accuracy_score_to_user = None
        self.model = None
        self.name = None


    # Logistic regression
    def logistic_regression(self):
        """LOGISTICAL REGRESSION, param grid: Logistic_regression__solver': ['liblinear', 'saga','lbfgs']"""
        from sklearn.linear_model import LogisticRegression
        self.name = "logistic_regression"
        LoR_model = LogisticRegression()
        operations = [('scaler',self.scaler),('Logistic_regression',LoR_model)]
        scaled_LoR_model_pipe = Pipeline(operations)
        param_grid = {'Logistic_regression__solver': ['liblinear', 'saga','lbfgs']}
        LoR_best_model = GridSearchCV(scaled_LoR_model_pipe,
                                      param_grid=param_grid,
                                      cv=10,
                                      scoring="accuracy")
        
        LoR_best_model.fit(self.X_train, self.y_train)
        self.model = LoR_best_model
        self.best_model_params = LoR_best_model.best_params_
        self.y_pred = LoR_best_model.predict(self.X_test)

    # KNN
    def knn_method(self):
        """KNN, param grid: k_values = list(range(1,30)), 'knn__n_neighbors':k_values, """
        from sklearn.neighbors import KNeighborsClassifier
        self.name = "knn"
        knn_model = KNeighborsClassifier()
        operations = [('scaler',self.scaler),('knn',knn_model)]
        pip_std_knn_class = Pipeline(operations)
        k_values = list(range(1,30))
        grid_param = {'knn__n_neighbors':k_values} 
        knn_best_model = GridSearchCV(pip_std_knn_class,
                              cv=10,
                              scoring="accuracy", 
                              param_grid=grid_param) 
        knn_best_model.fit(self.X_train, self.y_train)
        self.model = knn_best_model
        self.best_model_params = knn_best_model.best_params_
        self.y_pred = knn_best_model.predict(self.X_test)

    # SVC
    def svc_method(self):
        """SVC, param grid: 'svc__C': np.logspace(0, 1, 10),  
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'svc__degree': np.arange(1, 9),  
            'svc__gamma': ['scale', 'auto'] """
        from sklearn.svm import SVC
        self.name = "svc"
        svc_model = SVC()
        operations = [('scaler',self.scaler),('svc',svc_model)]
        pip_std_svc_class = Pipeline(operations)
        param_grid = {
            'svc__C': np.logspace(0, 1, 10),  
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'svc__degree': np.arange(1, 9),  
            'svc__gamma': ['scale', 'auto'] 
        }
        svc_model_best_model = GridSearchCV(pip_std_svc_class,
                                param_grid=param_grid,
                                cv=10,
                                scoring="accuracy")
        
        svc_model_best_model.fit(self.X_train,self.y_train)
        self.model = svc_model_best_model
        self.best_model_params = svc_model_best_model.best_params_
        self.y_pred = svc_model_best_model.predict(self.X_test)

    # REPORT ALL MODELS
    def report_cost_c(self):
        """Confusion matrix, Classification report, Accuracy"""
        confusion_matrix_to_user = confusion_matrix(self.y_test,self.y_pred)
        classification_report_to_user = classification_report(self.y_test,self.y_pred)
        self.accuracy_score_to_user = accuracy_score(y_true=self.y_test,y_pred=self.y_pred)
        return f"confusion_matrix\n{confusion_matrix_to_user}\n\nclassification_report\n{classification_report_to_user}\n\n"\
        f"Best Params\n{self.best_model_params}\n\n accuracy score\n{self.accuracy_score_to_user}"\
        f"\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    
    def send_accuracy_score(self):
        return self.accuracy_score_to_user
    
    def dump_best_model(self):
        from joblib import dump
        final_model = self.model.fit(self.X,self.y)
        dump(final_model, f"final_model_{self.name}.joblib")





    

