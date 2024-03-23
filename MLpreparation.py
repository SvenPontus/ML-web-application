import pandas as pd
from mlclass import MLR, MLC
import numpy as np


class MLpreparation:
    """This class is made for pandas operations and a mix of other stuff"""
    DF_LIST = list() 
    BEST_MODEL = list()
    FINAL_MODEL = list()

    def __init__(self, df=None):
        self.df = df
        
    # method to read many sorts of csv files
    def read_csv_pandas(self):
        encodings = ['utf-8', 'ISO-8859-1', 'latin1']
        delimiters = [',', ';', '|']
        error_messages = []

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    # Throw away self.df.seek(0)
                    self.df = pd.read_csv(self.df, encoding=encoding, delimiter=delimiter)
                    MLpreparation.DF_LIST.append(self.df)
                    return self.df
                except Exception as e:
                    error_messages.append(f"Failed with encoding {encoding} and delimiter '{delimiter}': {e}")

        raise ValueError(f"Unable to read the file with the provided encodings and delimiters. Errors: {error_messages}")
    
    # To basic csv side page 
    def info_about_df(self):
        df = self.df
        
        #Check non numeric value
        non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        non_numeric_columns_to_user = ""
        if non_numeric_columns:
            for _ in non_numeric_columns:
                non_numeric_columns_to_user += _ + "\n"
        else:
            non_numeric_columns_to_user += "0"
        # send strings of rows, features, filesize and which features are objects
        return f"<br>Rows : {len(df)}"\
                f"<br>Features : {len(df.columns)}"\
                f"<br>Size : {(df.memory_usage().sum() / 1024).round(1)} KB"\
                f"<br>Objects : {non_numeric_columns_to_user}"

    @staticmethod    
    def do_dummies(user_y_label):
        # Pick up the latest df from user
        df = MLpreparation.DF_LIST[-1]

        # Target Label from user
        y_label_name = df.columns[user_y_label]
        y = df[[y_label_name]]  

        # Drop y label to get X features
        X = df.drop(y_label_name, axis=1)

        # Process non-numeric columns
        non_numeric_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        for col in non_numeric_columns:
            unique_values = X[col].nunique()
            if unique_values == 2:
                X = pd.get_dummies(X, columns=[col], drop_first=True)

        df_dummies = pd.concat([y, X], axis=1)
        MLpreparation.DF_LIST.append(df_dummies)
    

    # Check users data regressor or classifier, to terminal app
    @staticmethod
    def controll_reg_or_cat(target_value,r_or_c):
        df = MLpreparation.DF_LIST[-1]
        column_name = df.columns[target_value]  
        column_dtype = df[column_name].dtype 
        if r_or_c == "r":
            # np.number have every int and float, I hope.. not shure 
            if np.issubdtype(column_dtype, np.number):
                return f"It is a continuous value."
            else:
                return f"ERROR! It is not a continuous value"

        # Check for classification task
        elif r_or_c == "c":
            if not np.issubdtype(column_dtype, np.number):
                return f"It is a categorical value."
            else:
                return f"ERROR! It is not a categorical value"
            

    # Controll nan values and if the csv file is ready for ml class
    @staticmethod
    def ready_for_ml_and_dummies(nr_user_y_label):
        # Pick up the latest df from user
        df = MLpreparation.DF_LIST[-1]
        # Target Label from user
        y_label_name = df.columns[nr_user_y_label]
        # Drop y label, for get X features
        X = df.drop(y_label_name, axis=1)

        messages = []  # To user
        dummies = []

        # Check for NaN values
        nan_values = [col for col in X.columns if X[col].isna().sum() > 0]
        if nan_values:
            nan_columns_formatted = ", ".join(nan_values)
            messages.append(f"Here you have your NaN values in columns: {nan_columns_formatted}.")

        # if non-numeric columns
        non_numeric_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        for col in non_numeric_columns:
            unique_values = X[col].nunique()
            if unique_values > 2:
                messages.append(f"Column {col} is not ready for dummies or ML. It has {unique_values} unique values.")
            elif unique_values == 2:
                dummies.append(f"You should convert {col} to dummy variables. It has {unique_values} unique values.")

        # If the file is ready then appends the message
        if not non_numeric_columns and not nan_values:
            messages.append("Your file is ready for ML.")

        # Return a single string with all messages
        return messages, dummies
    
            
    # Send the latest uploaded csv file like a df
    @staticmethod
    def send_df():
        if MLpreparation.DF_LIST:
            return MLpreparation.DF_LIST[-1]
        return None
    
    # several step in this method, choose target, split data and call ml class
    @staticmethod
    def pick_up_target_split_and_call_ml(nr_user_y_label, r_or_c):
        if r_or_c == "r":
            # Pick up latest df from user
            df = MLpreparation.DF_LIST[-1]
            # Target Label from user
            y_label_name = df.columns[nr_user_y_label]
            # Split data
            y = df[y_label_name]
            X = df.drop(y_label_name, axis=1)
            ###############################################################################
            # MLR class for all mlr models, look ml_class_pipeline for more understanding #
            ###############################################################################
            # Linear Regression
            LiR_mlr = MLR(y=y,X=X)
            LiR_mlr.linear_regression()
            LiR_mlr_cost = LiR_mlr.report_cost_func()
            LiR_r2_score = LiR_mlr.send_model_r2_score()
            # Lasso
            lasso_mlr = MLR(y=y,X=X)
            lasso_mlr.lasso_method()
            lasso_mlr_cost = lasso_mlr.report_cost_func()
            lasso_r2_score = lasso_mlr.send_model_r2_score()
            # Ridge
            ridge_mlr = MLR(y=y,X=X)
            ridge_mlr.ridge_method()
            ridge_mlr_cost = ridge_mlr.report_cost_func()
            ridge_r2_score = ridge_mlr.send_model_r2_score()
            # Elasticnet
            elasticnet_mlr = MLR(y=y,X=X)
            elasticnet_mlr.ridge_method()
            elasticnet_mlr_cost = elasticnet_mlr.report_cost_func()
            elasticnet_r2_score = elasticnet_mlr.send_model_r2_score()
            # SVR
            svr_mlr = MLR(y=y,X=X)
            svr_mlr.svr_method()
            svr_mlr_cost = svr_mlr.report_cost_func()
            svr_r2_score = svr_mlr.send_model_r2_score()
            
            models_r2_scores = {
                'Linear Regression': LiR_r2_score,
                'Lasso': lasso_r2_score,
                'Ridge': ridge_r2_score,
                'ElasticNet': elasticnet_r2_score,
                'SVR': svr_r2_score
            }
            # Find the best models_r2_score
            best_model_name = max(models_r2_scores, key=models_r2_scores.get)
            best_r2_score = models_r2_scores[best_model_name]
            best_model_and_score = f"{best_model_name} {best_r2_score.round(3)}"


            # Send best model to list, then dump when user want
            if best_model_name == 'Linear Regression':
                MLpreparation.FINAL_MODEL.append(LiR_mlr)
            elif best_model_name == 'Lasso':
                MLpreparation.FINAL_MODEL.append(lasso_mlr)
            elif best_model_name == 'Ridge':
                MLpreparation.FINAL_MODEL.append(ridge_mlr)
            elif best_model_name == 'ElasticNet':
                MLpreparation.FINAL_MODEL.append(elasticnet_mlr)
            elif best_model_name == 'SVR':
                MLpreparation.FINAL_MODEL.append(svr_mlr)

            MLpreparation.BEST_MODEL.append(best_model_and_score)
            
            # Return all models cost functions like strings
            return f"\nLinear Regression\n{LiR_mlr_cost}\n\nLasso\n{lasso_mlr_cost}\n\n"\
            f"Ridge\n{ridge_mlr_cost}\n\nElasticNet\n{elasticnet_mlr_cost}\n\n"\
            f"SVR\n{svr_mlr_cost}\n\n"            

        elif r_or_c == "c":
            df = MLpreparation.DF_LIST[-1]
            y_label_name = df.columns[nr_user_y_label]
            y = df[y_label_name]
            X = df.drop(y_label_name, axis=1)
            ################################
            # MLC class for all mlc models #
            ################################
            # logistic regression
            LoR_mlc = MLC(y=y,X=X)
            LoR_mlc.logistic_regression()
            LoR_mlc_cost = LoR_mlc.report_cost_c()
            LoR_accuracy_score = LoR_mlc.send_accuracy_score()
            # Knn
            knn_mlc = MLC(y=y,X=X)
            knn_mlc.knn_method()
            knn_mlc_cost = knn_mlc.report_cost_c()
            knn_mlc_accuracy_score = knn_mlc.send_accuracy_score()
            # SVC
            svc_mlc = MLC(y=y,X=X)
            svc_mlc.svc_method()
            svc_mlc_cost = svc_mlc.report_cost_c()
            svc_accuracy_score = svc_mlc.send_accuracy_score()
            
            accuracy_score_c = {
                'Logistic Regression': LoR_accuracy_score,
                'KNN': knn_mlc_accuracy_score,
                'SVC': svc_accuracy_score,
            }
            best_model_name = max(accuracy_score_c, key=accuracy_score_c.get)
            best_accuracy_score_c = accuracy_score_c[best_model_name]
            best_model_and_score = f"{best_model_name} {best_accuracy_score_c.round(2)}"

            if best_model_name == 'Logistic Regression':
                MLpreparation.FINAL_MODEL.append(LoR_mlc)
            elif best_model_name == 'KNN':
                MLpreparation.FINAL_MODEL.append(knn_mlc)
            elif best_model_name == 'SVC':
                MLpreparation.FINAL_MODEL.append(svc_mlc)

            MLpreparation.BEST_MODEL.append(best_model_and_score)

            return f"\nLogistic regression \n{LoR_mlc_cost}\n\n"\
            f"KNN\n{knn_mlc_cost}\n\nSVC\n{svc_mlc_cost}"

    # Create the fina model       
    @staticmethod
    def dump_best_model_final():
       final_model = MLpreparation.FINAL_MODEL[-1]
       final_model.dump_best_model()
        
    # Return best model
    @staticmethod
    def send_best_model_score():
        return f"{MLpreparation.BEST_MODEL[-1]} "


