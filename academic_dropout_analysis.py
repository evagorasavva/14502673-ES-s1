import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import joblib
import logging
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats

class AcademicDropoutAnalysis:
    def __init__(self, data_path, delimiter=',', random_state=42, num_features=5):
        self.random_state = random_state
        self.delimiter = delimiter
        self.num_features = num_features
        self.best_params = {}
        self.setup_logging()
        self.logger.info("Initializing Academic Dropout Analysis pipeline")
        self.load_data(data_path)
        self.perform_eda()
        self.detect_and_handle_outliers()
        self.prepare_data()
        self.identify_features()
        self.create_preprocessing_pipeline()
        self.split_data()
        self.train_and_evaluate_models()
        self.tune_models()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'academic_dropout_analysis_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path):
        """Load data and perform initial cleanup."""
        try:
            self.data = pd.read_csv(data_path, sep=self.delimiter, engine='python')
            # Clean column names: lower, strip spaces and quotes
            self.data.columns = self.data.columns.str.lower().str.strip().str.replace('"', '')
            self.logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            self.logger.info(f"Columns in the dataset: {self.data.columns.tolist()}")

            # Drop columns with >50% missing values
            missing_thresh = len(self.data) * 0.5
            initial_columns = self.data.shape[1]
            self.data = self.data.dropna(axis=1, thresh=missing_thresh)
            dropped_columns = initial_columns - self.data.shape[1]
            self.logger.info(f"Dropped {dropped_columns} columns with >50% missing values")

         
            self._remove_constant_features()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _remove_constant_features(self):
        """Remove constant and quasi-constant features."""
        nunique = self.data.nunique()
        constant_features = nunique[nunique == 1].index.tolist()
        quasi_constant_features = []  
        self.data.drop(columns=constant_features + quasi_constant_features, inplace=True)
        
        self.logger.info(f"Removed {len(constant_features)} constant features")

    def perform_eda(self):
        """Perform Exploratory Data Analysis (EDA)."""
        eda_dir = 'eda'
        os.makedirs(eda_dir, exist_ok=True)

        self.logger.info("Starting Exploratory Data Analysis (EDA)")

        # Identify the target column
        target_col = 'target'
        if target_col not in self.data.columns:
            self.logger.error("No 'Target' column found in dataset. Ensure the target column is named 'Target'.")
            raise ValueError("Target column ('Target') not found.")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target from numeric for plotting distributions
        numeric_cols = [col for col in numeric_cols if col != target_col]
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Function to create a safe filename
        def safe_filename(name):
            return name.replace('/', '_').replace('\\', '_').replace(' ', '_')

        # Plot distribution of numerical features
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            filename = f'distribution_{safe_filename(col)}.png'
            plt.savefig(os.path.join(eda_dir, filename))
            plt.close()

        # Plot count of categorical features (limit to top 10 categories)
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            top_categories = self.data[col].value_counts().nlargest(10)
            sns.barplot(x=top_categories.values, y=top_categories.index)
            plt.title(f'Count of Top 10 Categories in {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            filename = f'count_{safe_filename(col)}.png'
            plt.savefig(os.path.join(eda_dir, filename))
            plt.close()

        # Correlation matrix for numerical features
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.savefig(os.path.join(eda_dir, 'correlation_matrix.png'))
            plt.close()

        # Class distribution plot for Target
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.data[target_col])
        plt.title('Class Distribution of Target')
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.savefig(os.path.join(eda_dir, 'class_distribution.png'))
        plt.close()

        self.logger.info("EDA completed and plots saved in 'eda/' directory")

    def detect_and_handle_outliers(self, z_score_threshold=3, outlier_ratio_threshold=0.1):
        """Detect and handle outliers in numerical features using a Z-score method.
        """
        self.logger.info("Starting outlier detection and handling")
        outlier_dir = 'outliers'
        os.makedirs(outlier_dir, exist_ok=True)

        target_col = 'target'
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target variable from outlier detection
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        features_to_drop = []
        for col in numeric_cols:
            # Drop Nan values for calculation
            col_data = self.data[col].dropna()
            if col_data.empty:
                continue  

            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(col_data))
            outliers = z_scores > z_score_threshold
            num_outliers = np.sum(outliers)
            outlier_ratio = num_outliers / len(col_data)

            self.logger.info(f"Feature '{col}': Found {num_outliers} potential outliers out of {len(col_data)} (ratio={outlier_ratio:.2f})")

            if outlier_ratio_threshold is not None and outlier_ratio > outlier_ratio_threshold:
                # Drop the entire feature if it has too many outliers
                self.logger.info(f"Dropping feature '{col}' due to excessive outliers (ratio={outlier_ratio:.2f} > {outlier_ratio_threshold})")
                features_to_drop.append(col)
            else:
                # Cap the outliers at 1st and 99th percentiles
                if num_outliers > 0:
                    lower_bound = self.data[col].quantile(0.01)
                    upper_bound = self.data[col].quantile(0.99)
                    self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                    self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
                    self.logger.info(f"Capped outliers in '{col}' to 1st and 99th percentiles")

                    # Plot boxplot after handling outliers
                    plt.figure(figsize=(8, 4))
                    sns.boxplot(x=self.data[col])
                    plt.title(f'Boxplot of {col} After Outlier Handling')
                    plt.savefig(os.path.join(outlier_dir, f'boxplot_{col}.png'))
                    plt.close()

        # Drop features identified for removal
        if features_to_drop:
            self.data.drop(columns=features_to_drop, inplace=True)
            self.logger.info(f"Dropped {len(features_to_drop)} features due to outlier issues: {features_to_drop}")

        self.logger.info("Outlier detection and handling completed. Boxplots saved in 'outliers/' directory")

    def prepare_data(self):
            """Prepare features and target variable."""
            # Identify target
            target_col = 'target'
            if target_col not in self.data.columns:
                self.logger.error("No 'Target' column found in dataset.")
                raise ValueError("Target column ('Target') not found.")

            # Encode the target since it's multiclass categorical (dropout, enrolled, graduate)
            self.target_encoder = LabelEncoder()
            self.data[target_col] = self.target_encoder.fit_transform(self.data[target_col])
            self.logger.info(f"Encoded target classes: {self.target_encoder.classes_}")

            # Separate features and target
            self.features = self.data.drop(columns=[target_col])
            self.target = self.data[target_col]
            self.logger.info("Features and target prepared.")

    def identify_features(self):
        """Identify numerical and categorical features."""
        
        self.numeric_features = self.features.select_dtypes(include=[np.number]).columns.tolist()
        # Identify categorical features as object or category
        self.categorical_features = self.features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Identified {len(self.numeric_features)} numerical features: {self.numeric_features}")
        self.logger.info(f"Identified {len(self.categorical_features)} categorical features: {self.categorical_features}")

    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline."""
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        self.logger.info("Preprocessing pipeline created")

    def split_data(self):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.target
        )
        
        self.logger.info(f"Split data into training ({X_train.shape}) and testing ({X_test.shape}) sets")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate_models(self):
        """Train and evaluate models."""
        # Defines the models and hyperparameter grids
        self.models = {
'LogisticRegression': {
    'model': LogisticRegression(
        max_iter=1000,
        random_state=self.random_state,
        multi_class='ovr'
    ),
    'params': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    }
},
'SVC': {
    'model': SVC(
        probability=True,
        random_state=self.random_state,
        decision_function_shape='ovr'
    ),
    'params': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['rbf', 'linear']
    }
},
'XGBoost': {
    'model': XGBClassifier(
        eval_metric='mlogloss',
        random_state=self.random_state,
    ),
    'params': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0]
    }
},
'GradientBoosting': {
    'model': GradientBoostingClassifier(
        random_state=self.random_state
    ),
    'params': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__subsample': [0.7, 0.8, 1.0]
    }
},
'KNeighbors': {
    'model': KNeighborsClassifier(),
    'params': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }
}
            }


        # Create directories for results and feature importance
        os.makedirs('results', exist_ok=True)
        feature_importance_dir = 'feature_importance'
        os.makedirs(feature_importance_dir, exist_ok=True)
        
        self.results = []
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, config in self.models.items():
            self.logger.info(f"\nTraining and evaluating {name}")
            pipeline = ImbPipeline([
                ('preprocessing', self.preprocessor),
                ('feature_selection', SelectKBest(score_func=f_classif, k=min(self.num_features, len(self.numeric_features) + len(self.categorical_features)))),
                ('smote', SMOTE(random_state=self.random_state)),
                ('classifier', config['model'])
            ])
            
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config['params'],
                scoring='f1_weighted',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.logger.info(f"{name} Grid Search completed")
            self.logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation F1 score for {name}: {grid_search.best_score_:.3f}")
            
            self.best_params[name] = grid_search.best_params_
            
            joblib.dump(grid_search.best_estimator_, os.path.join('results', f'{name}_best_pipeline.joblib'))
            self.logger.info(f"Saved {name} best pipeline")

            y_pred = grid_search.predict(self.X_test)

            # For multiclass ROC AUC, need predict_proba
            if hasattr(grid_search.best_estimator_.named_steps['classifier'], "predict_proba"):
                y_pred_proba = grid_search.predict_proba(self.X_test)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                # If no predict_proba, skip ROC AUC or sets to None
                roc_auc = None
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            self.logger.info(f"{name} Test Accuracy: {accuracy:.3f}")
            self.logger.info(f"{name} Test F1 Score: {f1:.3f}")
            if roc_auc is not None:
                self.logger.info(f"{name} Test ROC AUC: {roc_auc:.3f}")
            else:
                self.logger.info(f"{name} Test ROC AUC: N/A (no predict_proba)")

            self.results.append({
                'Model': name,
                'CV_F1_Mean': round(grid_search.best_score_, 3),
                'Test_Accuracy': round(accuracy, 3),
                'Test_F1': round(f1, 3),
                'Test_ROC_AUC': round(roc_auc, 3) if roc_auc is not None else None
            })
            
            with open(os.path.join('results', f'{name}_report.txt'), 'w') as f:
                f.write(f"Classification Report for {name}:\n")
                f.write(classification_report(self.y_test, y_pred, target_names=self.target_encoder.classes_))
                f.write(f"\nBest cross-validation F1 score: {grid_search.best_score_:.3f}")
                f.write(f"\nBest parameters: {grid_search.best_params_}")
            
            # Feature importance
            if name in ['RandomForest', 'GradientBoosting', 'XGBoost','SVC']:
                try:
                    preprocessor = grid_search.best_estimator_.named_steps['preprocessing']
                    # Attempt to get feature names
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        all_feature_names = preprocessor.get_feature_names_out()
                    else:
                        #if can't get names
                        all_feature_names = self.numeric_features + self.categorical_features
                    
                    feature_selector = grid_search.best_estimator_.named_steps['feature_selection']
                    if hasattr(feature_selector, 'get_support'):
                        selected_indices = feature_selector.get_support(indices=True)
                        selected_features = [all_feature_names[i] for i in selected_indices]
                    else:
                        selected_features = all_feature_names
                    
                    classifier = grid_search.best_estimator_.named_steps['classifier']
                    if hasattr(classifier, 'feature_importances_'):
                        importances = classifier.feature_importances_
                    elif hasattr(classifier, 'coef_'):
                        importances = np.mean(np.abs(classifier.coef_), axis=0)
                    else:
                        self.logger.warning(f"No feature importances for {name}.")
                        continue
                    
                    if len(importances) != len(selected_features):
                        self.logger.warning("Mismatch in importances and features length.")
                        continue
                    
                    fi_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
                    plt.title(f'Feature Importances for {name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(feature_importance_dir, f'feature_importance_{name}.png'))
                    plt.close()
                    
                    fi_df.to_csv(os.path.join(feature_importance_dir, f'feature_importance_{name}.csv'), index=False)
                    self.logger.info(f"Feature importance analysis completed for {name}.")
                except Exception as e:
                    self.logger.error(f"Error during feature importance analysis for {name}: {e}")
                    continue

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join('results', 'model_performance_initial.csv'), index=False)
        self.logger.info("Initial model performance metrics saved.")

    def tune_models(self):
        """Tune the models further to achieve better performance using refined hyperparameter grids."""
        self.logger.info("\nStarting the tuning of models for better performance.")
        
        for name, config in self.models.items():
            self.logger.info(f"\nTuning {name}")
            try:
                best_pipeline_path = os.path.join('results', f'{name}_best_pipeline.joblib')
                if not os.path.exists(best_pipeline_path):
                    self.logger.warning(f"Best pipeline for {name} not found. Skipping tuning for this model.")
                    continue
                pipeline = joblib.load(best_pipeline_path)
                
                initial_best_params = self.best_params.get(name, {})
                
                refined_params = {}
                for param, value in initial_best_params.items():
                    # Only refine if value is numeric 
                    if 'subsample' in param or 'colsample_bytree' in param:
                        base_values = [value]
                        #Decreasing/increasing slightly if it does not exceed [0,1]
                        for delta in [-0.05, 0.05]:
                            new_val = value + delta
                            if 0 < new_val <= 1:
                                base_values.append(new_val)
                        # Ensures unique and sorted
                        base_values = sorted(set(base_values))
                        refined_params[param] = base_values
                    elif isinstance(value, (int, float)):
                        # adjusted by +/- 1 
                        candidates = [max(value-1, 1), value, value+1]
                        # Remove duplicates and ensure positive values if needed
                        candidates = [c for c in candidates if c > 0]
                        refined_params[param] = sorted(set(candidates))
                    else:
                        # Non-numeric or parameters that shouldn't be refined
                        refined_params[param] = [value]

                # Remove parameters that ended up empty
                refined_params = {k: v for k, v in refined_params.items() if len(v) > 0}
                
                # If no refined parameters, skip tuning
                if not refined_params:
                    self.logger.warning(f"No refined parameters defined for {name}. Skipping tuning.")
                    continue
                
                self.logger.info(f"Refined hyperparameter grid for {name}: {refined_params}")
                
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=refined_params,
                    scoring='f1_weighted',
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                self.logger.info(f"{name} Refined Grid Search completed")
                self.logger.info(f"Best parameters after tuning for {name}: {grid_search.best_params_}")
                self.logger.info(f"Best cross-validation F1 score after tuning for {name}: {grid_search.best_score_:.3f}")
                
                joblib.dump(grid_search.best_estimator_, os.path.join('results', f'{name}_tuned_pipeline.joblib'))
                self.logger.info(f"Saved {name} tuned pipeline to 'results/{name}_tuned_pipeline.joblib'")
                
                y_pred = grid_search.predict(self.X_test)
                if hasattr(grid_search.best_estimator_.named_steps['classifier'], "predict_proba"):
                    y_pred_proba = grid_search.predict_proba(self.X_test)
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = None
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.logger.info(f"{name} Tuned Test Accuracy: {accuracy:.3f}")
                self.logger.info(f"{name} Tuned Test F1 Score: {f1:.3f}")
                if roc_auc is not None:
                    self.logger.info(f"{name} Tuned Test ROC AUC: {roc_auc:.3f}")
                else:
                    self.logger.info(f"{name} Tuned Test ROC AUC: N/A")
                
                self.results.append({
                    'Model': f"{name}_Tuned",
                    'CV_F1_Mean': round(grid_search.best_score_, 3),
                    'Test_Accuracy': round(accuracy, 3),
                    'Test_F1': round(f1, 3),
                    'Test_ROC_AUC': round(roc_auc, 3) if roc_auc is not None else None
                })
                
                with open(os.path.join('results', f'{name}_tuned_report.txt'), 'w') as f:
                    f.write(f"Classification Report for {name} Tuned Model:\n")
                    f.write(classification_report(self.y_test, y_pred, target_names=self.target_encoder.classes_))
                    f.write(f"\nBest cross-validation F1 score after tuning: {grid_search.best_score_:.3f}")
                    f.write(f"\nBest parameters after tuning: {grid_search.best_params_}")
                
            except Exception as e:
                self.logger.error(f"Error during tuning for {name}: {e}")
                continue
        
        tuned_results_df = pd.DataFrame([res for res in self.results if 'Tuned' in res['Model']])
        if not tuned_results_df.empty:
            tuned_results_df.to_csv(os.path.join('results', 'model_performance_tuned.csv'), index=False)
            self.logger.info("Tuned model performance metrics saved.")

def main():
    parser = argparse.ArgumentParser(description='Academic Dropout Analysis Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--delimiter', type=str, default=';', help='Delimiter used in the data file')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--num_features', type=int, default=36, help='Number of top features to select')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    analysis = AcademicDropoutAnalysis(
        data_path=args.data_path,
        delimiter=args.delimiter,
        random_state=args.random_state,
        num_features=args.num_features
    )

if __name__ == "__main__":
    main()