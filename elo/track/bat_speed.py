import os

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import plot_importance
import joblib
import xgboost as xgb

from database_driver import DatabaseDriver


class Driver:
    def __init__(self, db_file, radar_table_name):
        self.input_variables_df = []
        self.database_driver = DatabaseDriver ()
        self.sigma = np.array([2.02, 1.50, 2.20])

    def retrieve_variables (self, table = 'variables'):
        self.input_variables_df = self.database_driver.read_variable_data (table)
        # self.input_variables_df = self.input_variables_df.drop_duplicates(subset='PitchUID', keep='first')
        # print (self.input_variables_df)
        # exit (0)
        # self.database_driver.write_data(self.input_variables_df, 'variable')
        self.input_variables_df = self.input_variables_df.drop_duplicates (subset = 'PitchUID', keep = 'first')
        self.input_variables_df = self.input_variables_df.dropna (subset = 'ExitSpeed')

    def is_barreled(self, row):
        speed = row['ExitSpeed']
        angle = row['Angle']

        if speed < 98:
            return 0
        elif 98 <= speed <= 99:
            return int(26 <= angle <= 30)
        elif 99 < speed < 116:
            lower_bound = max(26 - (speed - 98), 8)
            upper_bound = min(30 + (speed - 98) * 2, 50)
            return int(lower_bound <= angle <= upper_bound)
        else: # speed >= 116
            return int(8 <= angle <= 50)

        # Apply the function to each row
        # df['Barrel'] = df.apply(is_barreled, axis=1)

    def prepare_data(self, batter_df, min):
        # Sorting by EV
        batter_df_sorted = batter_df.sort_values(by='ExitSpeed', ascending=False)
        # Selecting top and bottom 15
        top_15 = batter_df_sorted.head(15)
        bottom_15 = batter_df_sorted.tail(15)
        # Combining them
        regression_df = pd.concat([top_15, bottom_15])
        regression_df['eA'] = np.where(regression_df.index.isin(top_15.index), 0.21, min)
        # print (regression_df.to_string ())
        # exit (0)
        return regression_df[['PitchUID', 'ExitSpeed', 'eA']]

    def apply_regression(self, batter_df, min):
        regression_data = self.prepare_data(batter_df, min)
        X = regression_data[['ExitSpeed']]  # Predictor variable
        y = regression_data['eA']  # Response variable

        # Linear Regression
        model = LinearRegression()
        model.fit(X, y)

        # Using the model to predict eA
        batter_df['eA'] = model.predict(batter_df[['ExitSpeed']])
        merged_df = batter_df.merge(regression_data [['PitchUID', 'eA']], on='PitchUID', how='left', suffixes=('', '_new'))
        merged_df['eA'] = merged_df['eA_new'].fillna(merged_df['eA'])
        merged_df.drop(columns=['eA_new'], inplace=True)
        batter_df = merged_df
        # print (batter_df.to_string ())
        # exit (0)

        return batter_df

    def approximate_barrel_ea (self):
        self.input_variables_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') |
            (self.input_variables_df['PitchCall'] == 'Foul')
            ]
        self.input_variables_df = self.input_variables_df [self.input_variables_df['ExitSpeed'] > 50]
        self.input_variables_df = self.input_variables_df [self.input_variables_df['RelSpeed'] > 70]
        self.input_variables_df ['Barrel'] = self.input_variables_df.apply(self.is_barreled, axis = 1)
        # filtered_df = self.input_variables_df.groupby('Batter').filter(lambda x: len(x) >= 100)
        input_variables_df = self.input_variables_df [self.input_variables_df['Barrel'] == 1]
        filtered_df = input_variables_df.groupby('Batter').filter(lambda x: len(x) >= 3)
        input_variables_df = pd.concat([self.apply_regression(batter_df, 0.19) for _, batter_df in tqdm(filtered_df.groupby('Batter'), desc="Processing batters")])
        input_variables_df.rename (columns={'eA': 'BarrelEA'}, inplace=True)
        print (self.input_variables_df)
        try:
            self.input_variables_df = self.input_variables_df.drop (['BarrelEA_x', 'BarrelEA_y'])
        except:
            print ('whoops barrel x y')
        try:
            self.input_variables_df = self.input_variables_df.drop (['BarrelEA'])
        except:
            print ('whoops barrel')
        self.input_variables_df = self.input_variables_df.merge (input_variables_df [['PitchUID', 'BarrelEA']], on = 'PitchUID',  how='left')
        self.input_variables_df.drop_duplicates('PitchUID')
        # print (self.input_variables_df)
        # exit (0)
        # self.input_variables_df = self.apply_regression (self.input_variables_df)

    def approximate_ea (self):
        self.input_variables_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') |
            (self.input_variables_df['PitchCall'] == 'Foul')
            ]
        self.input_variables_df = self.input_variables_df [self.input_variables_df['ExitSpeed'] > 50]
        self.input_variables_df = self.input_variables_df [self.input_variables_df['RelSpeed'] > 70]
        filtered_df = self.input_variables_df.groupby('Batter').filter(lambda x: len(x) >= 70)
        self.input_variables_df = pd.concat([self.apply_regression(batter_df, -0.1) for _, batter_df in tqdm(filtered_df.groupby('Batter'), desc="Processing batters")])
        # self.input_variables_df = self.apply_regression (self.input_variables_df)


    def commit_variables (self):
        self.database_driver.write_data (self.input_variables_df, 'batting_variables')

    def calculate_vbat(row):
        return (row['EV'] - row['eA'] * row['vball']) / (1 + row['eA'])
    def calculate_bat_speeds (self):
        self.input_variables_df ['BatSpeed'] = (self.input_variables_df['ExitSpeed'] - self.input_variables_df['eA'] * self.input_variables_df['ZoneSpeed']) / (1 + self.input_variables_df['eA'])
        self.input_variables_df ['BarrelBatSpeed'] = (self.input_variables_df['ExitSpeed'] - self.input_variables_df['BarrelEA'] * self.input_variables_df['ZoneSpeed']) / (1 + self.input_variables_df['BarrelEA'])
        self.input_variables_df ['BarrelBatSpeed'].fillna(value=np.nan)

    def calculate_hand_barrel_speeds(row):
        print ('hi')

    def calculate_SF (self):
        try:
            cols_to_drop = [col for col in self.input_variables_df.columns if "AverageSF" in col]
            self.input_variables_df = self.input_variables_df.drop(columns=cols_to_drop)
        except:
            print ('whoops')
        self.input_variables_df ['SF'] = 1 + (self.input_variables_df['ExitSpeed'] - self.input_variables_df['BatSpeed'])/(self.input_variables_df['ZoneSpeed'] + self.input_variables_df ['BatSpeed'])
        variables_df = self.database_driver.read_variable_data ('variables')
        variables_df.dropna (subset = 'ExitSpeed')
        variables_df = variables_df.merge(self.input_variables_df[['PitchUID', 'SF']],
                                       on='PitchUID', how='left')

        # Fill missing values in the 'Target' column with 0
        variables_df['SF'] = variables_df['SF'].fillna(0)
        variables_df = variables_df[
            (variables_df['PitchCall'] == 'InPlay') |
            (variables_df['PitchCall'] == 'StrikeSwinging') |
            (variables_df['PitchCall'] == 'Foul')
            ]
        # variables_df = variables_df [variables_df['Batter'] == 'Ahuna, Maui']
        variables_df.loc[variables_df['PitchCall'] == 'Foul', 'SF'] = 0
        variables_df['AverageSF'] = variables_df.groupby('Batter')['SF'].transform('mean')
        # print (variables_df.to_string ())
        # exit (0)
        self.input_variables_df = self.input_variables_df.merge(variables_df[['PitchUID', 'AverageSF']],
                                          on='PitchUID', how='left')

    def calculate_averages (self):
        input_variables_df = self.input_variables_df [self.input_variables_df ['ExitSpeed'] > 50]
        self.input_variables_df['AverageBatSpeed'] = input_variables_df.groupby('Batter')['BatSpeed'].transform('mean')
        self.input_variables_df['TrueBatSpeed'] = input_variables_df.groupby('Batter')['BarrelBatSpeed'].transform('mean')
        self.input_variables_df['AverageHandSpeed'] = input_variables_df.groupby('Batter')['HandSpeed'].transform('mean')
        self.input_variables_df['AverageBarrelSpeed'] = input_variables_df.groupby('Batter')['BarrelSpeed'].transform('mean')
        self.input_variables_df['AverageEA'] = self.input_variables_df.groupby('Batter')['eA'].transform('mean')
        self.input_variables_df['AverageI'] = self.input_variables_df.groupby('Batter')['I'].transform('mean')

    def find_intrinsic_values (self):
        # Vectorized Gaussian kernel function
        def gaussian_kernel(x, X, sigma):
            # No need to explicitly calculate diffs as a separate array
            # Use broadcasting to calculate scaled_diffs and exp_component in a memory-efficient manner
            scaled_diffs = (X - x) / sigma
            exp_component = np.exp(-0.5 * np.sum(scaled_diffs ** 2, axis=1))
            coefficient = 1 / ((2 * np.pi) ** (1.5) * np.prod(sigma))
            return coefficient * exp_component

        def kde(X, x_new, sigma, batch_size=1000):
            # Initialize an empty list to store densities
            all_densities = []
            # Process in batches to reduce memory usage
            for start_idx in tqdm(range(0, len(x_new), batch_size)):
                end_idx = start_idx + batch_size
                batch_x_new = x_new[start_idx:end_idx]
                # Compute densities for the current batch
                batch_densities = np.array([gaussian_kernel(x, X, sigma) for x in batch_x_new])
                all_densities.append(batch_densities)
            # Concatenate all batch densities and calculate the mean density for each x_new
            densities = np.concatenate(all_densities, axis=0)
            return np.mean(densities, axis=1)

        # Estimate P(Rj | x) for each x_new in a vectorized manner
        def estimate_probabilities(X, Y, X_new, sigma, outcomes):
            # print ('A')
            p_x = kde(X, X_new, sigma)
            probabilities = []

            for outcome in tqdm (outcomes):
                X_rj = X[Y == outcome]
                p_x_given_rj = kde(X_rj, X_new, sigma)
                p_rj = len(X_rj) / len(X)

                # Apply Bayes' theorem: P(Rj | x) = (p(x | Rj) * P(Rj)) / p(x)
                p_rj_given_x = (p_x_given_rj * p_rj) / p_x

                probabilities.append(p_rj_given_x)

            # Normalize probabilities to ensure they sum to 1 across all outcomes for each x_new
            total_probabilities = np.sum(probabilities, axis=0)
            normalized_probabilities = probabilities / total_probabilities

            return normalized_probabilities

        # Usage
        batting_variables_df = self.input_variables_df [['ExitSpeed', 'Angle', 'Direction', 'PlayResult']]
        value_map = {
            "Fielder's Choice" : 'Out',
            'Sacrifice' : 'Out',
            'sacrifice' : 'Out'
        }
        batting_variables_df['PlayResult'] = batting_variables_df['PlayResult'].replace(value_map)
        batting_variables_df = batting_variables_df[batting_variables_df['PlayResult'].isin(['Out', 'Single', 'Double', 'Triple', 'HomeRun'])]
        X = batting_variables_df[['ExitSpeed', 'Angle', 'Direction']].to_numpy()
        Y = batting_variables_df['PlayResult'].to_numpy()
        X_new = X
        outcomes = np.unique(Y)
        normalized_probabilities = estimate_probabilities(X, Y, X_new, self.sigma, outcomes)

        # Print the probabilities
        for index, x_new in enumerate(X_new):
            print(f"Estimated P(Rj | x) for {x_new}:")
            for outcome_idx, outcome in enumerate(outcomes):
                print(f"{outcome}: {normalized_probabilities[outcome_idx][index]}")
            print("\n")

    def calculate_percentiles (self):
        players_df = self.input_variables_df [['Batter', 'BatterTeam', 'BatterSide', 'TrueBatSpeed', 'AverageBatSpeed', 'AverageSF', 'SwingDecision', 'AverageEA', 'AverageI', 'NeutralExitSpeed', 'NeutralHR']]
        # print (players_df.head ().to_string ())
        players_df = players_df.drop_duplicates(subset=['Batter'])
        def calculate_percentiles(df):
            # Group the DataFrame by PitchType and then apply the percentile function
            percentiles_df = df.transform(lambda x: round(100*x.rank(pct=True),0))#if np.issubdtype(x.dtype, np.number) else x)
            # Keep non-numeric columns as they are
            for col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    if col != 'TrueBatSpeed':
                        percentiles_df[col] = df[col]
            return percentiles_df

        players_percentiles_df = calculate_percentiles(players_df)
        # print (players_percentiles_df)
        self.database_driver.write_data (players_percentiles_df, 'Percentiles_Batters')
        # print (percentiles_df.to_string ())

    def train_classifier (self):
        # features = ['ExitSpeed', 'Angle', 'Direction']
        features = ['PlateLocHeight', 'PlateLocSide']
        batting_variables_df = self.input_variables_df
        # print (batting_variables_df.head ().to_string ())
        batting_variables_df = batting_variables_df [['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PlayResult']]
        df2 = self.database_driver.read_variable_data('variables')
        df2 = df2 [['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PitchCall']]
        df2.rename(columns={'PitchCall': 'PlayResult'}, inplace=True)
        # print (df2 ['PlayResult'].unique ())
        df2 = df2[df2['PlayResult'] == 'StrikeSwinging']
        # print (df2)
        df2 = df2[df2['Batter'].isin(batting_variables_df['Batter'])]
        # print (df2)
        columns_to_append = df2[['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PlayResult']]
        print (columns_to_append)
        batting_variables_df = pd.concat([batting_variables_df, columns_to_append], ignore_index=True)
        batting_variables_df = batting_variables_df [batting_variables_df ['PlayResult'] != 'StolenBase']
        batting_variables_df = batting_variables_df [batting_variables_df ['PlayResult'] != 'CaughtStealing']
        batting_variables_df = batting_variables_df [batting_variables_df ['BatterSide'] == 'Right']
        # print (batting_variables_df)
        value_map = {
            'Undefined' : 'Foul',
            "FieldersChoice" : 'Out',
            'Sacrifice' : 'Out',
            'sacrifice' : 'Out',
            'OUt' : 'Out',
            'Error' : 'Out'
        }
        batting_variables_df['PlayResult'] = batting_variables_df['PlayResult'].replace(value_map)
        X = batting_variables_df [features]
        y = batting_variables_df['PlayResult']
        value_map = {
            'StrikeSwinging' : 0,
            'Foul' : 1,
            'Out' : 2,
            'Single' : 3,
            'Double' : 4,
            'Triple' : 5,
            'HomeRun' : 6
        }
        y = y.replace (value_map)
        # print(y.unique())

        print (batting_variables_df['PlayResult'].unique ())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        def objective(trial):
            estimators = trial.suggest_int('n_estimators', 150, 800)
            stopping_rounds = estimators // 10
            param = {
                'verbosity': 0,
                # 'objective': 'binary:logistic',
                # 'eval_metric': 'logloss',
                'n_estimators': estimators,
                'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'early_stopping_rounds': stopping_rounds,
                'objective': 'multi:softprob',
                'num_class': 7,
                'eval_metric': 'mlogloss',
                # 'enable_categorical': True
            }

            clf = xgb.XGBClassifier(**param, enable_categorical=True)
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            # if (self.multi):
            preds = clf.predict_proba(X_test)
            # else:
            #     preds = clf.predict_proba(X_test)[:, 1]
            trial.set_user_attr("model", clf)
            binary_preds = np.where(preds > 0.5, 1, 0)
            # score = f1_score(y_test, binary_preds)
            # score = roc_auc_score(y_test, preds)
            # class_preds = np.argmax(preds, axis=1)
            score = log_loss(y_test, preds)
            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=80)
        best_model = study.best_trial.user_attrs["model"]
        best_params = study.best_trial.params
        model_directory = "Full_Model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Save the best model to a file within the 'Model' directory
        model_filename = os.path.join(model_directory, f'joblib_model_R_loc.json')
        # model_filename = f'{self.focus.name}_{self.currently_modeling}-{self.current_pitch_class}.json' # or use .bin for binary format
        best_model.save_model(model_filename)

        model_directory = "JobLib_Model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename = os.path.join(model_directory, f'joblib_model_R_loc.joblib')
        # Save the best model to a file
        joblib.dump(best_model, model_filename)
        # Optionally, save the best parameters to a file as well
        with open(f'best_params_R_loc.txt', 'w') as f:
            f.write(str(best_params))
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        print ('Result:', study.best_value)
        clf = study.best_trial.user_attrs["model"]
        feature_importances = clf.get_booster().get_score(importance_type='weight')
        print("Feature Importances:")
        for feature, importance in feature_importances.items():
            print(f"{feature}: {importance}")
        plt.figure(figsize=(12, 10))
        plot_importance(clf, importance_type='weight', max_num_features=10)  # Show top 10 features
        plt.show()
        # self.current_df['PitchType'] = self.current_df['PitchType'].astype('category')
        # self.current_df['PitcherThrows'] = self.current_df['PitcherThrows'].astype('category')
        # self.current_df['BatterSide'] = self.current_df['BatterSide'].astype('category')
        # self.input_variables_df['PredictedClass'] = clf.predict(self.input_variables_df[features])
        # self.input_variables_df['PredictedProbability'] = clf.predict_proba(self.input_variables_df[features])[:, 1]
        # class_labels = clf.classes_
        # probabilities = clf.predict_proba(self.input_variables_df[features])
        # Create a new column for each class probability
        # for i, class_label in enumerate(class_labels):
        #     self.input_variables_df[f'Prob_{class_label}'] = probabilities[:, i]

    def predict_intrinsic_values (self):
        model_filename = f'JobLib_Model/joblib_model_intrinsic_value.joblib'
        # self.model.load_model (model_filename)
        clf = joblib.load (model_filename)
        features = ['ExitSpeed', 'Angle', 'Direction']
        class_labels = clf.classes_
        probabilities = clf.predict_proba(self.input_variables_df[features])
        # Create a new column for each class probability
        for i, class_label in enumerate(class_labels):
            self.input_variables_df[f'Prob_{class_label}'] = probabilities[:, i]

    def predict_loc_values (self):
        batting_variables_df = self.input_variables_df
        # print (batting_variables_df.head ().to_string ())
        # batting_variables_df = batting_variables_df [['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PlayResult']]
        df2 = self.database_driver.read_variable_data('variables')
        # df2 = df2 [['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PitchCall']]
        # df2.rename(columns={'PitchCall': 'PlayResult'}, inplace=True)
        # print (df2 ['PlayResult'].unique ())
        # df2 = df2[df2['PlayResult'] == 'StrikeSwinging']
        # print (df2)
        df2 = df2[df2['Batter'].isin(batting_variables_df['Batter'])]
        # print (df2)
        # columns_to_append = df2[['Batter', 'BatterSide', 'PlateLocHeight', 'PlateLocSide', 'PlayResult']]
        # print (columns_to_append)
        # batting_variables_df = pd.concat([batting_variables_df, columns_to_append], ignore_index=True)

        model_R = joblib.load('JobLib_Model/joblib_model_R_loc.joblib')
        model_L = joblib.load('JobLib_Model/joblib_model_L_loc.joblib')

        # Features to use for predictions
        features = ['PlateLocHeight', 'PlateLocSide']

        # Iterate through the DataFrame and make predictions based on BatterSide
        df_left = df2[df2['BatterSide'] == 'Left']
        df_right = df2[df2['BatterSide'] != 'Left']

        # Function to add prediction probabilities to DataFrame
        def add_predictions(df, model, features):
            # Ensure the input is in the correct shape for prediction
            X = df[features].values
            # Predict probabilities
            probabilities = model.predict_proba(X)
            class_labels = model.classes_

            # Create a DataFrame with the probabilities
            prob_df = pd.DataFrame(probabilities, columns=[f'Prob_loc_{a}' for a in class_labels], index=df.index)

            # Concatenate the probabilities back to the original DataFrame
            return pd.concat([df, prob_df], axis=1)

        # Apply the function to both subsets
        df_left_predicted = add_predictions(df_left, model_L, features)
        df_right_predicted = add_predictions(df_right, model_R, features)

        # Concatenate the results back into one DataFrame
        df2 = pd.concat([df_left_predicted, df_right_predicted]).sort_index()
        df2 = self.calculate_I (df2, 'loc_I')
        self.database_driver.write_data(df2, 'variable')
        print (df2.head ().to_string ())
    def calculate_row_I(self, row, foul_values, starting_values):
        balls = row['Balls']  # Assuming 'balls' is a column name
        strikes = row['Strikes']  # Assuming 'strikes' is a column name
        foul_value = foul_values.get((balls, strikes), 'default')
        starting_value = starting_values.get((balls, strikes), 'default')

        # Compute 'I' based on the logic provided
        I = (
                row['Prob_0'] * foul_value
                + row['Prob_1'] * (0 - starting_value)
                + row['Prob_2'] * (0.892 - starting_value)
                + row['Prob_3'] * (1.283 - starting_value)
                + row['Prob_4'] * (1.635 - starting_value)
                + row['Prob_5'] * (2.135 - starting_value)  # Assuming you meant 'Prob_5' instead of the second 'Prob_4'
        )
        return I
    def calculate_row_I_swing(self, row, foul_values, starting_values, strike_values):
        balls = row['Balls']  # Assuming 'balls' is a column name
        strikes = row['Strikes']  # Assuming 'strikes' is a column name
        foul_value = foul_values.get((balls, strikes), 'default')
        strike_value = strike_values.get((balls, strikes), 'default')
        starting_value = starting_values.get((balls, strikes), 'default')

        # Compute 'I' based on the logic provided
        I = (
                row['Prob_loc_0'] * strike_value
                + row['Prob_loc_1'] * foul_value
                + row['Prob_loc_2'] * (0 - starting_value)
                + row['Prob_loc_3'] * (0.892 - starting_value)
                + row['Prob_loc_4'] * (1.283 - starting_value)
                + row['Prob_loc_5'] * (1.635 - starting_value)
                + row['Prob_loc_6'] * (2.135 - starting_value)  # Assuming you meant 'Prob_5' instead of the second 'Prob_4'
        )
        return I
    def calculate_I (self, df = None, name = 'I'):
        # strike_value = -0.093
        starting_values = {
            (0, 0): 0.310,
            (0, 1): 0.262,
            (0, 2): 0.196,
            (1, 0): 0.355,
            (1, 1): 0.293,
            (1, 2): 0.223,
            (2, 0): 0.436,
            (2, 1): 0.352,
            (2, 2): 0.273,
            (3, 0): 0.622,
            (3, 1): 0.472,
            (3, 2): 0.384,
        }
        # foul_values = {
        #     (0, 0): -0.093,
        #     (0, 1): -0.097,
        #     # (0, 2): -0.223,
        #     (0, 2): 0,
        #     (1, 0): -0.143,
        #     (1, 1): -0.129,
        #     # (1, 2): -0.273,
        #     (1, 2): 0,
        #     (2, 0): -0.270,
        #     (2, 1): -0.197,
        #     # (2, 2): -0.384,
        #     (2, 2): -0.384,
        #     (3, 0): -0.219,
        #     (3, 1): -0.305,
        #     # (3, 2): -0.689,
        #     (3, 2): 0
        # }
        # strike_values = {
        #     (0, 0): -0.093,
        #     (0, 1): -0.097,
        #     (0, 2): -0.223,
        #     (1, 0): -0.143,
        #     (1, 1): -0.129,
        #     (1, 2): -0.273,
        #     (2, 0): -0.270,
        #     (2, 1): -0.197,
        #     (2, 2): -0.384,
        #     (3, 0): -0.219,
        #     (3, 1): -0.305,
        #     (3, 2): -0.689,
        # }
        strike_values = {
            (0, 0): -0.048,
            (0, 1): -0.066,
            (0, 2): -0.196,
            (1, 0): -0.062,
            (1, 1): -0.070,
            (1, 2): -0.223,
            (2, 0): -0.084,
            (2, 1): -0.079,
            (2, 2): -0.273,
            (3, 0): -0.152,
            (3, 1): -0.086,
            (3, 2): -0.384,
        }
        foul_values = {
            (0, 0): -0.048,
            (0, 1): -0.066,
            (0, 2): 0,
            (1, 0): -0.062,
            (1, 1): -0.070,
            (1, 2): 0,
            (2, 0): -0.084,
            (2, 1): -0.079,
            (2, 2): 0,
            (3, 0): -0.152,
            (3, 1): -0.086,
            (3, 2): 0,
        }
        tqdm.pandas(desc="Calculating values")
        if (df is None):
            self.input_variables_df[name] = self.input_variables_df.progress_apply(
                lambda row: self.calculate_row_I(row, foul_values, starting_values), axis=1
            )
        else:
            df[name] = df.progress_apply(
                lambda row: self.calculate_row_I_swing(row, foul_values, starting_values, strike_values), axis=1
            )
            return df

    def calculate_take_I (self):
        df2 = self.database_driver.read_table_data('variable')
        # balls = df2['Balls']  # Assuming 'balls' is a column name
        # strikes = df2['Strikes']  # Assuming 'strikes' is a column name
        # strike_values = {
        #     (0, 0): -0.093,
        #     (0, 1): -0.097,
        #     (0, 2): -0.223,
        #     (1, 0): -0.143,
        #     (1, 1): -0.129,
        #     (1, 2): -0.273,
        #     (2, 0): -0.270,
        #     (2, 1): -0.197,
        #     (2, 2): -0.384,
        #     (3, 0): -0.219,
        #     (3, 1): -0.305,
        #     (3, 2): -0.689,
        # }
        strike_values = {
            (0, 0): -0.048,
            (0, 1): -0.066,
            (0, 2): -0.196,
            (1, 0): -0.062,
            (1, 1): -0.070,
            (1, 2): -0.223,
            (2, 0): -0.084,
            (2, 1): -0.079,
            (2, 2): -0.273,
            (3, 0): -0.152,
            (3, 1): -0.086,
            (3, 2): -0.384,
        }
        ball_values = {
            (0, 0): 0.045,
            (0, 1): 0.031,
            (0, 2): 0.027,
            (1, 0): 0.081,
            (1, 1): 0.059,
            (1, 2): 0.050,
            (2, 0): 0.186,
            (2, 1): 0.118,
            (2, 2): 0.111,
            (3, 0): 0.067,
            (3, 1): 0.219,
            (3, 2): 0.305,
        }
        df2['strike_value'] = df2.apply(lambda row: strike_values.get((row['Balls'], row['Strikes']), 'default'), axis=1)
        df2['ball_value'] = df2.apply(lambda row: ball_values.get((row['Balls'], row['Strikes']), 'default'), axis=1)

        # Condition to apply to each row for the 'take_loc_I' column
        conditions = (
                (df2['PlateLocHeight'] < 3.6) &
                (df2['PlateLocHeight'] > 1.5) &
                (df2['PlateLocSide'] > -17/2) &
                (df2['PlateLocSide'] < 17/2)
        )
        # Apply conditions to set 'take_loc_I' value
        df2['take_loc_I'] = df2['strike_value']
        df2.loc[~conditions, 'take_loc_I'] = df2['ball_value']#-df2['strike_value']
        df2 = self.calculate_I(df2, 'loc_I')
        df2['swing_difference'] = df2 ['loc_I'] - df2 ['take_loc_I']
        value_map = {
            "StrikeSwinging" : 1,
            "SwingingStrike" : 1,
            'StrikeTaken' : 0,
            'StrikeCalled' : 0,
            'Ball' : 0,
            'InPlay' : 1,
            'Foul' : 1,
            'HitByPitch' : 0,
            'BallIntentional' : 0
        }
        df2['Swing'] = df2['PitchCall'].replace(value_map)
        # df2['Swing'] = df2['PitchCall'].replace(value_map)
        self.database_driver.write_data(df2, 'variable')
        # print (df2.head ().to_string ())

    def train_classifier_swing_diff (self):
        # features = ['ExitSpeed', 'Angle', 'Direction']
        features = ['Balls', "Strikes", 'swing_difference']
        batting_variables_df = self.database_driver.read_variable_data('variable')
        X = batting_variables_df [features]
        y = batting_variables_df['Swing']
        print(y.unique())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        def objective(trial):
            estimators = trial.suggest_int('n_estimators', 150, 800)
            stopping_rounds = estimators // 10
            param = {
                'verbosity': 0,
                # 'objective': 'binary:logistic',
                # 'eval_metric': 'logloss',
                'n_estimators': estimators,
                'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'early_stopping_rounds': stopping_rounds,
                'objective': 'binary:logistic',
                # 'num_class': 2,
                'eval_metric': 'logloss',
                # 'enable_categorical': True
            }

            clf = xgb.XGBClassifier(**param, enable_categorical=True)
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            # if (self.multi):
            preds = clf.predict_proba(X_test)
            # else:
            #     preds = clf.predict_proba(X_test)[:, 1]
            trial.set_user_attr("model", clf)
            binary_preds = np.where(preds > 0.5, 1, 0)
            # score = f1_score(y_test, binary_preds)
            # score = roc_auc_score(y_test, preds)
            # class_preds = np.argmax(preds, axis=1)
            score = log_loss(y_test, preds)
            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=60)
        best_model = study.best_trial.user_attrs["model"]
        best_params = study.best_trial.params
        model_directory = "Full_Model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Save the best model to a file within the 'Model' directory
        model_filename = os.path.join(model_directory, f'joblib_model_swing_diff.json')
        # model_filename = f'{self.focus.name}_{self.currently_modeling}-{self.current_pitch_class}.json' # or use .bin for binary format
        best_model.save_model(model_filename)

        model_directory = "JobLib_Model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename = os.path.join(model_directory, f'joblib_model_swing_diff.joblib')
        # Save the best model to a file
        joblib.dump(best_model, model_filename)
        # Optionally, save the best parameters to a file as well
        with open(f'best_params_swing_diff.txt', 'w') as f:
            f.write(str(best_params))
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        print ('Result:', study.best_value)
        clf = study.best_trial.user_attrs["model"]
        feature_importances = clf.get_booster().get_score(importance_type='weight')
        print("Feature Importances:")
        for feature, importance in feature_importances.items():
            print(f"{feature}: {importance}")
        plt.figure(figsize=(12, 10))
        plot_importance(clf, importance_type='weight', max_num_features=10)  # Show top 10 features
        plt.show()

    def predict_swing (self):
        df2 = self.database_driver.read_variable_data('variable')
        # df2.drop (['Prob_swing_0', 'Prob_swing_1'], axis = 1)
        model_filename = f'JobLib_Model/joblib_model_swing_diff.joblib'
        # self.model.load_model (model_filename)
        clf = joblib.load (model_filename)
        features = ['Balls', "Strikes", 'swing_difference']
        class_labels = clf.classes_
        probabilities = clf.predict_proba(df2[features])
        # Create a new column for each class probability
        for i, class_label in enumerate(class_labels):
           df2[f'Prob_swing_{class_label}'] = probabilities[:, i]
        self.database_driver.write_data(df2, 'variable')
    def calculate_credit (self):
        df2 = self.database_driver.read_variable_data('variable')
        df2 ['adj_Prob_Swing'] = np.minimum (df2 ['Prob_swing_1'], 0.95) - 0.5
        df2 ['Credit'] = df2 ['adj_Prob_Swing'] * (df2 ['Swing'] * 2 - 1)
        self.database_driver.write_data(df2, 'variable')

    def aggregate_credit (self):
        df2 = self.database_driver.read_variable_data('variable')
        try:
            self.input_variables_df.drop (['SwingDecision'], axis = 1)
            self.input_variables_df.drop (['Credit_x'], axis = 1)
            self.input_variables_df.drop (['Credit_y'], axis = 1)
        except:
            print ('whoops')
        average_credit = df2.groupby('Batter')['Credit'].mean().reset_index()
        self.input_variables_df = pd.merge(self.input_variables_df, average_credit, on='Batter', how='left')
        self.input_variables_df.rename(columns={'Credit': 'SwingDecision'}, inplace=True)
        # print (self.input_variables_df)
    def calculate_neutral_speed (self):
        df = self.input_variables_df
        df['diff_from_0.1'] = (df['eA'] - 0.1).abs()
        top_15_closest = df.sort_values(['Batter', 'diff_from_0.1']).groupby('Batter').head(15)
        average_exit_speed = top_15_closest.groupby('Batter')['ExitSpeed'].mean().reset_index()
        average_exit_speed.rename(columns={'ExitSpeed': 'NeutralExitSpeed'}, inplace=True)
        df_merged = pd.merge(df, average_exit_speed, on='Batter', how='left')
        self.input_variables_df = df_merged.drop(columns=['diff_from_0.1'])
    def calculate_neutral_HR (self):
        df = self.input_variables_df
        # df.drop (['NeutralHRPower'], axis = 1)
        # df.rename(columns={'Prob_5_x': 'Prob_5'}, inplace=True)
        df['diff_from_0.1'] = (df['eA'] - 0.1).abs()
        top_15_closest = df.sort_values(['Batter', 'diff_from_0.1']).groupby('Batter').head(15)
        average_exit_speed = top_15_closest.groupby('Batter')['Prob_5'].mean().reset_index()
        average_exit_speed.rename(columns={'Prob_5': 'NeutralHR'}, inplace=True)
        df_merged = pd.merge(df, average_exit_speed, on='Batter', how='left')
        self.input_variables_df = df_merged.drop(columns=['diff_from_0.1'])
        # print (df_merged)
    def add_columns (self, columns, table):
        self.database_driver.add_column(columns, table)
    def calculate_hand_speeds (self):
        df = self.input_variables_df
        arm_length = 18
        distance_from_hands = 25
        bat_length = 34
        # df ['PointOfContact'] = (df ['eA'] + 0.2 )/0.4 * (bat_length - 6) + arm_length + distance_from_hands
        df ['PointOfContact'] = arm_length + distance_from_hands
        df ['HandSpeed'] = df ['BarrelBatSpeed'] / df ['PointOfContact']  * (arm_length)
        df ['BarrelSpeed'] = df ['BarrelBatSpeed'] / df ['PointOfContact'] * (arm_length + bat_length)
        # df.loc[(df['eA'] < -0.1) | (df['eA'] > 0.1), 'HandSpeed'] = np.nan
        # df.loc[(df['eA'] > 0.1), 'HandSpeed'] = np.nan
        # df.loc[(df['eA'] < -0.1), 'BarrelSpeed'] = np.nan
        # df.loc[(df['eA'] > 0.1) | (df['eA'] < -0.1), 'BarrelSpeed'] = np.nan
        self.input_variables_df = df
        # self.input_variables_df = pd.merge (self.input_variables_df, df, on = 'PitchUID')
        # print (self.input_variables_df)
        # exit (0)
    def calculate_attack_angle(self):
        # Assuming 'df' is your DataFrame
        df = self.input_variables_df.copy()
        bin_size = 3  # Adjust bin size as needed
        # df['LA_Bin'] = pd.cut(df['Angle'], bins=np.arange(df['Angle'].min(), df['Angle'].max() + bin_size, bin_size))
        df['LA_Bin'] = pd.cut(df['Angle'], bins=np.arange(-30, 55, bin_size))
        # Placeholder for peak exit velocity results
        batter_peak_ev = {}

        # Iterate over each batter in the dataset
        for batter in tqdm(df['Batter'].unique()):
            batter_df = df[df['Batter'] == batter]

            def select_top_bbe(x):
                top_fraction = 0.2  # Selecting the top 20% of BBEs
                num_top_bbe = max(int(len(x) * top_fraction), 1)  # Ensure at least 1 BBE is selected
                return x.nlargest(num_top_bbe, 'ExitSpeed')

            # Filter Top BBEs by Exit Velocity in Each Bin for the batter
            top_bbe_df = batter_df.groupby('LA_Bin', group_keys=False).apply(select_top_bbe)

            # Proceed if there are enough points to fit a parabola
            if len(top_bbe_df) >= 3:
                # Fit the parabola
                def parabola(x, a, b, c):
                    return a * x**2 + b * x + c
                try:
                    params, _ = curve_fit(parabola, top_bbe_df['Angle'], top_bbe_df['ExitSpeed'])
                    a, b, c = params
                    h = -b / (2 * a)
                    # k = a * h**2 + b * h + c
                except:
                    # In case the curve fitting fails, set a default value
                    h = np.nan
            else:
                h = np.nan

            batter_peak_ev[batter] = h
        print(batter_peak_ev)
        # Map the peak exit velocity results back to the original DataFrame
        self.input_variables_df['AttackAngle'] = self.input_variables_df['Batter'].map(batter_peak_ev)
        # print(self.input_variables_df)
        # exit(0)


    def write_percentages (self):
        self.database_driver.write_percentages_batter()

# print (df_final)

driver = Driver ('radar2.db', 'radar_data')
# driver.retrieve_variables()
# driver.approximate_ea()
# driver.commit_variables()
driver.retrieve_variables('batting_variables')
driver.calculate_attack_angle()
# driver.calculate_hand_speeds()
driver.commit_variables()
# driver.approximate_ea()
# driver.commit_variables()
# driver.approximate_barrel_ea ()
# driver.commit_variables()
# driver.retrieve_variables('batting_variables')
# # driver.commit_variables()
# driver.calculate_bat_speeds()
# driver.commit_variables()
# driver.calculate_hand_speeds()
# driver.commit_variables()
# driver.commit_variables()
# driver.calculate_SF()
# driver.calculate_averages()
# driver.commit_variables()
# driver.commit_variables()
# driver.commit_variables()
# driver.train_classifier()
# driver.predict_intrinsic_values()
# driver.calculate_I()
# driver.predict_loc_values()
# driver.calculate_take_I()
# driver.train_classifier_swing_diff()
# driver.predict_swing()
# driver.calculate_credit()
# driver.aggregate_credit()
# driver.commit_variables()
# driver.calculate_I()
# driver.commit_variables()
# driver.calculate_neutral_speed()
# driver.calculate_neutral_HR()
# driver.commit_variables()
# driver.calculate_percentiles()
driver.write_percentages()
# driver.retrieve_variables('batting_variables')
# driver.find_intrinsic_values()

def process_data ():
    driver.retrieve_variables()
    driver.approximate_ea()
    driver.commit_variables()
    driver.add_columns (['PitchUID', 'Angle', 'Direction'], 'batting_variables')
    driver.retrieve_variables('batting_variables')
    driver.approximate_barrel_ea ()
    driver.commit_variables()
    driver.retrieve_variables('batting_variables')
    driver.calculate_bat_speeds()
    driver.calculate_hand_speeds()
    driver.calculate_SF()
    driver.predict_intrinsic_values()
    driver.calculate_I()
    driver.commit_variables()
    driver.predict_loc_values()
    driver.calculate_take_I()
    driver.predict_swing()
    driver.calculate_credit()
    driver.aggregate_credit()
    driver.calculate_neutral_speed()
    driver.calculate_neutral_HR()
    driver.calculate_averages()
    driver.commit_variables()
    driver.calculate_attack_angle()
    driver.commit_variables()
    driver.calculate_percentiles()
    driver.write_percentages()
    # driver.retrieve_variables('batting_variables')

# process_data()