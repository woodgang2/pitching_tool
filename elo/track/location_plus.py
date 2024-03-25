import enum
import math
import os
import sqlite3
import seaborn as sns
import optuna
import xgboost as xgb
import joblib
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
#TODO: how identify Cutter_S when auto pitch classifies sinkers?
from flask import Flask, render_template, request, app
from scipy.stats import gaussian_kde
# from shiny import App, render, ui
import numpy as np
import pandas as pd
from keras.src.layers import Masking
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, log_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import coolwarm
# from tensorflow import keras
# keras = tf.keras
# import tensorflow.python.keras.api._v1.keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Dropout, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from xgboost import plot_importance

class Focus(enum.Enum):
    Stuff = 1
    Location = 2
    Overall = 3

def calculate_averages(row, averages_df):
    pitcher = row['Pitcher']
    if pitcher in averages_df.index:
        row['AverageFastballRS'] = averages_df.loc[pitcher, 'RelSpeed']
        row['AverageFastballIVB'] = averages_df.loc[pitcher, 'InducedVertBreak']
        row['AverageFastballHB'] = averages_df.loc[pitcher, 'HorzBreak']
    return row

class Driver:
    def __init__(self, db_file, radar_table_name, focus = Focus.Location):
        self.db_file = db_file
        self.table_name = radar_table_name
        self.radar_df = []
        self.input_variables_df = []
        self.features = []
        self.current_pitch_class = 'None'
        self.currently_modeling = 'None'
        self.focus = focus
        self.context_features = ['Balls', 'Strikes', 'PlateLocHeight', 'PlateLocSide']
        self.params = {}
        self.current_df = []
        self.multi = False
        self.predictions_df = []
        self.xRV_df = []
        self.players_df = []
        self.percentiles_df = []
        self.model = xgb.Booster()

    def read_radar_data (self, new = 0):
        print ("Reading radar data")
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {self.table_name}', conn).iloc[0, 0]

        # Choose a chunk size
        chunksize = 10000

        # Initialize a progress bar
        pbar = tqdm(total=total_rows)

        # Placeholder DataFrame
        df_list = []

        # Read data in chunks
        for chunk in pd.read_sql_query(f'SELECT * FROM {self.table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])

        # Concatenate all chunks into a single DataFrame
        self.radar_df = pd.concat(df_list, ignore_index=True)
        # print (self.radar_df)
        # numeric_series = pd.to_numeric(self.radar_df['HitTrajectoryXc8'], errors='coerce')

        # Count the number of non-NaN values (i.e., numeric values)
        # num_numeric_values = numeric_series.notna().sum()
        # print (num_numeric_values)
        # Close the progress bar
        pbar.close()

        # Close the database connection
        print (self.radar_df ['PitchCall'].unique ())
        value_map = {
            # 'Fastball': 'Four-Seam',
            # 'ChangeUp': 'ChangeUp',
            'FourSeamFastBall': 'Four-Seam',
            'TwoSeamFastBall': 'Sinker',
            'Undefined': 'Other',
            # 'Knuckleball': 'Other',
            'OneSeamFastball': 'Sinker',
            ',': 'Other'
            # Add other replacements here if necessary
        }
        if (new > 0):
            self.radar_df['PitchType'] = self.radar_df['TaggedPitchType'].replace(value_map)
            self.radar_df.to_sql ('radar_data', conn, if_exists='replace', index=False)
        print ('Finished reading radar data')
        # filtered_df = self.radar_df[self.radar_df['PitcherTeam'] == 'VIR_CAV']
        # filtered_df = filtered_df[['Pitcher']]
        # filtered_df.to_sql('UVA_Pitchers', conn, if_exists='replace', index=False)
        conn.close()

    def read_variable_data (self):
        print ("Reading variable data")
        table_name = 'variables'
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {self.table_name}', conn).iloc[0, 0]
        chunksize = 10000
        pbar = tqdm(total=total_rows)
        df_list = []
        for chunk in pd.read_sql_query(f'SELECT * FROM {table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])
        self.input_variables_df = pd.concat(df_list, ignore_index=True)
        pbar.close()
        # print (self.radar_df ['PitchCall'].unique ())
        value_map = {
            'FoulBallNotFieldable' : 'Foul',
            'FoulBallFieldable' : 'Foul',
            'FoulBall' : 'Foul',
            'WildPitch' : 'Ball',
            'SwinginStrike' : 'SwingingStrike',
            'StirkeCalled' : 'StrikeCalled',
            'BallinDirt' : 'Ball',
            'BallCalled' : 'Ball'
        }
        self.input_variables_df['PitchCall'] = self.input_variables_df['PitchCall'].replace(value_map)
        self.input_variables_df = self.input_variables_df[self.input_variables_df['PitchCall'] != 'Undefined']
        self.input_variables_df = self.input_variables_df[self.input_variables_df['PitchCall'] != 'CatchersInterference']
        self.input_variables_df = self.input_variables_df[self.input_variables_df['PitchCall'] != 'BattersInterference']
        value_map = {
            'Flyball' : 'FlyBall',
            'Popup' : 'FlyBall',
            'PopUp' : 'FlyBall',
            'Groundball' : 'GroundBall',
            'groundBall' : 'GroundBall'
        }
        self.input_variables_df['TaggedHitType'] = self.input_variables_df['TaggedHitType'].replace(value_map)
        # self.input_variables_df = self.input_variables_df[self.input_variables_df['TaggedHitType'] != 'Undefined']
        self.input_variables_df = self.input_variables_df[self.input_variables_df['TaggedHitType'] != 'Bunt']
        self.input_variables_df = self.input_variables_df[self.input_variables_df['TaggedHitType'] != ',']
        # self.input_variables_df = self.input_variables_df.loc[(self.input_variables_df['PitchType'] == 'Cutter') & (self.input_variables_df['DifferenceRS'] != 0), 'PitchType'] = 'Cutter_S'
        # self.input_variables_df.loc[(self.input_variables_df['PitchType'] == 'Cutter') & (self.input_variables_df['DifferenceRS'] != 0), 'PitchType'] = 'Cutter_S'
        # self.input_variables_df = self.input_variables_df.merge(
        #     self.radar_df[['PitchUID', 'AutoPitchType']],
        #     on='PitchUID',
        #     how='left'
        self.input_variables_df = self.input_variables_df.drop_duplicates(subset='PitchUID', keep='first')
        # self.input_variables_df.loc[(self.input_variables_df['PitchType'] == 'Fastball') & (self.input_variables_df['AutoPitchType'] != 'Changeup'), 'PitchType'] = self.input_variables_df['AutoPitchType']
        # self.input_variables_df.loc[(self.input_variables_df['PitchType'] == 'Fastball') & (self.input_variables_df['AutoPitchType'] == 'Changeup'), 'PitchType'] = 'Sinker'
        # self.input_variables_df = self.input_variables_df.drop ('AutoPitchType', axis = 1)
        print (self.input_variables_df ['PitchCall'].unique ())
        print (self.input_variables_df ['TaggedHitType'].unique ())
        print (self.input_variables_df ['PitchType'].unique ())

        # self.input_variables_df = pd.merge(self.input_variables_df, self.radar_df[['PitchUID'] + ['ZoneSpeed']], on='PitchUID', how='left')

        # print (self.input_variables_df)
        # self.radar_df.to_sql ('radar_data', conn, if_exists='replace', index=False)
        conn.close()

    def read_predictions (self, focus):
        print ("Reading predictions")
        table_name = f'{focus.name}_Probabilities'
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {table_name}', conn).iloc[0, 0]
        chunksize = 10000
        pbar = tqdm(total=total_rows)
        df_list = []
        for chunk in pd.read_sql_query(f'SELECT * FROM {table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])
        self.predictions_df = pd.concat(df_list, ignore_index=True)
        self.predictions_df = self.predictions_df.drop_duplicates(subset='PitchUID')
        pbar.close()
        conn.close()
        # self.write_predictions(Focus.Location)
        # self.predictions_df = self.predictions_df [self.predictions_df ["PitchType"] == "Four-Seam"]
        # sns.jointplot(data=self.predictions_df, x='AxisDifference', y='SpinEfficiency', kind='hex', gridsize=50, cmap='inferno', C=self.predictions_df['Prob_SwingingStrike'], marginal_kws=dict(bins=50, fill=True))
        # plt.colorbar(label='Prob_SwingingStrike').set_label(label='xWhiff%', size=20)
        # plt.show()
        # x = self.predictions_df ['']
        # fig, ax = plt.subplots()
        # AxisDifference = self.predictions_df ['AxisDifference']
        # SpinEfficiency = self.predictions_df ['SpinEfficiency']
        # Prob_SwingingStrike = self.predictions_df ['Prob_SwingingStrike']
        # hb = ax.hexbin(
        #     AxisDifference,
        #     SpinEfficiency,
        #     Prob_SwingingStrike,
        #     gridsize=10,
        #     mincnt=1,# Increased grid size for better resolution
        #     cmap='RdYlBu',  # Color map
        #     extent=[-50, 75, 0, 100],
        #     fill = True,
        # reduce_C_function=np.mean  # Use mean for the color intensity
        # )

        # Add a color bar to the plot
        # cb = plt.colorbar(hb, ax=ax)
        # cb.set_label('Prob_SwingingStrike')

        # Set axis labels
        # ax.set_xlabel('AxisDifference')
        # ax.set_ylabel('SpinEfficiency')

        # Show the plot
        # plt.show()

        # Display the plot
        # plt.show()
        # exit (0)

    def calculate_inferred_spin_axis (self):
        print ('Calculating Inferred Axis')
        # self.radar_df ['InferredSpinAxis'] = self.radar_df ['pfxz']/self.radar_df ['pdfxx']*180/math.pi + 90
        self.radar_df = self.radar_df.dropna(subset=['pfxz'])
        self.radar_df = self.radar_df.dropna(subset=['pfxx'])
        self.radar_df['InferredSpinAxis'] = np.where(self.radar_df['pfxx'] < 0,
                                                     (np.arctan(self.radar_df['pfxz'] / self.radar_df['pfxx']) * 180 / math.pi + 90) + 180,
                                                     np.arctan (self.radar_df['pfxz'] / self.radar_df['pfxx']) * 180 / math.pi + 90)
        # print (self.radar_df)

    def calculate_seam_shifted_wake (self):
        print ('Calculating Wake')
        try:
            self.radar_df.drop(['BasePositionY', 'BasePositionX', 'BasePositionX', 'ThrowTrajectoryZc2', 'ThrowTrajectoryZc1', 'ThrowTrajectoryZc0', 'ThrowTrajectoryYc0', 'ThrowTrajectoryYc1', 'ThrowTrajectoryYc2', 'ThrowTrajectoryXc0', 'ThrowTrajectoryXc1', 'ThrowTrajectoryXc2'], axis=1, inplace=True)
        except:
            print ('Already dropped rows')
        self.radar_df['AxisDifference'] = self.radar_df ['InferredSpinAxis'] - self.radar_df ['SpinAxis']
        self.radar_df['AxisDifference'] = np.where(np.abs(self.radar_df['AxisDifference']) > 180,
                                                   360 - np.abs(self.radar_df['AxisDifference']),
                                                   self.radar_df['AxisDifference'])
    def calculate_spin_efficiency (self):
        print ('Calculating Efficiency')
        ##yoinked this code from https://github.com/jmaschino56/BaseballSavantPitchProfiler/blob/master/PitchProfiler.py
        g_fts = 32.174
        R_ball = .121
        mass = 5.125
        circ = 9.125
        temp = 72
        humidity = 50
        pressure = 29.92
        temp_c = (5/9)*(temp-32)
        pressure_mm = (pressure * 1000) / 39.37
        svp = 4.5841 * math.exp((18.687 - temp_c/234.5) * temp_c/(257.14 + temp_c))
        rho = (1.2929 * (273 / (temp_c + 273)) * (pressure_mm - .3783 *
                                                  humidity * svp / 100) / 760) * .06261
        const = 0.07182 * rho * (5.125 / mass) * (circ / 9.125)**2
        spin_efficiencies = []
        spin_directions = []
        for i in tqdm(range(len(self.radar_df.PitchType))):
            v0 = self.radar_df.RelSpeed.iloc[i]
            vx0 = self.radar_df.vx0.iloc[i]
            ax = self.radar_df.ax0.iloc[i]
            vy0 = self.radar_df.vy0.iloc[i]
            ay = self.radar_df.ay0.iloc[i]
            vz0 = self.radar_df.vz0.iloc[i]
            az = self.radar_df.az0.iloc[i]
            pfx_x = self.radar_df.pfxx.iloc[i]
            pfx_z = self.radar_df.pfxz.iloc[i]
            plate_x = self.radar_df.PlateLocSide.iloc[i]
            plate_z = self.radar_df.PlateLocHeight.iloc[i]
            release_x = self.radar_df.RelSide.iloc[i]
            release_y = 60.5-self.radar_df.Extension.iloc[i]
            release_z = self.radar_df.RelHeight.iloc[i]
            spin_rate = self.radar_df.SpinRate.iloc[i]

            # time between release and y0 measurement
            t_back_to_release = (-vy0-math.sqrt(vy0**2-2*ay*(50-release_y)))/ay

            # adjust velocity at y0 to be at release
            vx_r = vx0+ax*t_back_to_release
            vy_r = vy0+ay*t_back_to_release
            vz_r = vz0+az*t_back_to_release
            dv0 = v0 - math.sqrt(vx_r**2 + vy_r**2 + vz_r**2)/1.467

            # calculate pitch time also know as tf in Template
            t_c = (-vy_r - math.sqrt(vy_r**2 - 2*ay*(release_y - 17/12))) / ay

            # calcualte x and z movement
            calc_x_mvt = (plate_x-release_x-(vx_r/vy_r)*(17/12-release_y))
            calc_z_mvt = (plate_z-release_z-(vz_r/vy_r)*(17/12-release_y))+0.5*g_fts*t_c**2

            # average velocity
            vx_bar = (2 * vx_r + ax * t_c) / 2
            vy_bar = (2 * vy_r + ay * t_c) / 2
            vz_bar = (2 * vz_r + az * t_c) / 2
            v_bar = math.sqrt(vx_bar**2 + vy_bar**2 + vz_bar**2)

            # drag acceleration
            adrag = -(ax * vx_bar + ay * vy_bar + (az + g_fts) * vz_bar)/v_bar

            # magnus acceleration
            amagx = ax + adrag * vx_bar/v_bar
            amagy = ay + adrag * vy_bar/v_bar
            amagz = az + adrag * vz_bar/v_bar + g_fts
            amag = math.sqrt(amagx**2 + amagy**2 + amagz**2)

            # movement components
            mx = .5 * amagx * (t_c**2)*12
            mz = .5 * amagz * (t_c**2)*12

            # drag/lift coefficients may need work
            Cd = adrag / (v_bar**2 * const)
            Cl = amag / (v_bar**2 * const)

            s = 0.4*Cl/(1-2.32*Cl)
            spin_t = 78.92*s*v_bar

            '''
            # for debugging purposes
            spin_tx = spin_t*(vy_bar*amagz-vz_bar*amagy)/(amag*v_bar)
            spin_ty = spin_t*(vz_bar*amagx-vx_bar*amagz)/(amag*v_bar)
            spin_tz = spin_t*(vx_bar*amagy-vy_bar*amagx)/(amag*v_bar)
            spin_check = math.sqrt(spin_tx**2+spin_ty**2+spin_tz**2)-spin_t
            '''
            # calc spin direction
            phi = 0
            if(amagz > 0):
                phi = math.atan2(amagz, -amagx) * 180/math.pi
            else:
                phi = 360+math.atan2(amagz, -amagx)*180/math.pi
            dec_time = 3-(1/30)*phi
            if(dec_time <= 0):
                dec_time += 12

            # calc spin eff
            spin_eff = spin_t/spin_rate
            spin_efficiencies.append(spin_eff)
            spin_directions.append (phi)
        self.radar_df['SpinEfficiency'] = spin_efficiencies
        self.radar_df['MagnusSpinAxis'] = spin_directions

    def normalize_VAA (self):
        # self.radar_df['VAA'] = self.radar_df['VertApprAngle'] / self.radar_df['PlateLocHeight']
        height_difference = self.radar_df['PlateLocHeight'] - 2.5
        vaa_adjustment = height_difference * 0.82  # Adjusting by 0.82 degrees for every foot
        self.radar_df ['VAA'] = self.radar_df['VertApprAngle'] - vaa_adjustment
    def classify_pitches (self):
        # print (self.input_variables_df)
        # self.input_variables_df = self.input_variables_df.drop (['Cluster1', 'Cluster2'], axis = 1)
        columns_to_drop = [col for col in self.input_variables_df.columns if "Cluster" in col]
        self.input_variables_df = self.input_variables_df.drop(columns=columns_to_drop)
        self.input_variables_df = self.input_variables_df.dropna(subset = ['AxisDifference', 'SpinEfficiency'])#, 'Cluster'])
        input_variables_df = self.input_variables_df [(self.input_variables_df ['PitchType'] == 'Four-Seam') | (self.input_variables_df ['PitchType'] == 'Sinker')]
        # pitch_uids = input_variables_df ['PitchUID']
        features = input_variables_df.drop(['RelSpeed', 'PitchType', 'PitchUID', 'PitcherTeam', 'PitchCall', 'TaggedHitType', 'ExitSpeed', 'PitcherThrows', 'BatterSide', 'Balls', 'Strikes', 'DifferenceIVB', 'DifferenceHB', 'Pitcher', 'RelHeight', 'RelSide', 'Extension', 'PlateLocHeight', 'PlateLocSide', 'SpinEfficiency', 'VAA'], axis = 1)
        print (features.head ().to_string ())
        # exit (0)
        # features = features.dropna()
        # self.input_variables_df = self.input_variables_df.dropna()
        # It's often a good practice to scale your features, especially for mixture models
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine the optimal number of components for GMM based on BIC
        # n_components = np.arange(1, 21)  # Considering 1 to 20 components
        n_components = 30
        # models = [GaussianMixture(n, covariance_type='full', random_state=0, verbose = 1).fit(features_scaled) for n in tqdm(n_components)]
        # bic_values = [model.bic(features_scaled) for model in tqdm(models)]
        #
        # # Plot the BIC scores
        # import matplotlib.pyplot as plt
        #
        # plt.plot(n_components, bic_values, marker='o')
        # plt.xlabel('Number of components')
        # plt.ylabel('BIC')
        # plt.show()
        #
        # # Select the model with the lowest BIC
        # optimal_n_components = n_components[np.argmin(bic_values)]

        # optimal_n_components = n_components
        # best_gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=0, verbose = 1).fit(features_scaled)
        #
        # # Predict the cluster assignment for each observation
        # cluster_assignments = best_gmm.predict(features_scaled)
        # bic_value1 = best_gmm.bic(features_scaled)
        # print (bic_value1)
        # joblib.dump(best_gmm, 'gmm_model.joblib')
        # # If you want to add these cluster assignments back to your original DataFrame
        # self.input_variables_df['Cluster1'] = cluster_assignments

        optimal_n_components = n_components + 1
        optimal_n_components = 2
        best_gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=0, verbose = 1).fit(features_scaled)

        # Predict the cluster assignment for each observation
        soft_probabilities = best_gmm.predict_proba(features_scaled)
        joblib.dump(best_gmm, f'gmm_model{optimal_n_components}.joblib')
        bic_value2 = best_gmm.bic(features_scaled)
        print (bic_value2)
        # If you want to add these cluster assignments back to your original DataFrame
        # self.input_variables_df['Cluster2'] = cluster_assignments
        probabilities_df = pd.DataFrame(soft_probabilities, columns=[f'Cluster_{i}' for i in range(optimal_n_components)])

        # If you want to add these probabilities to your original DataFrame, you can concatenate them
        # Assuming 'self.input_variables_df' is your original DataFrame
        # input_variables_df ['PitchUID'] = pitch_uids
        input_variables_df = pd.concat([input_variables_df, probabilities_df], axis=1)
        columns_to_add = [col for col in input_variables_df.columns if "Cluster" in col]
        # print (self.input_variables_df.head ().to_string ())
        # print (input_variables_df.head ().to_string ())
        # self.input_variables_df = self.input_variables_df.merge (input_variables_df [columns_to_add], on='PitchUID', how = 'left', axis =1)
        self.input_variables_df = pd.merge(self.input_variables_df, input_variables_df[['PitchUID'] + columns_to_add], on='PitchUID', how='left')

    def clear_fastball_averages (self):
        self.radar_df.drop(columns=['AverageSinkerRS', 'AverageSinkerHB', 'AverageSinkerIVB', 'AverageCutterRS', 'AverageCutterHB', 'AverageCutterIVB', 'AverageFour-SeamRS', 'AverageFour-SeamHB', 'AverageFour-SeamIVB'], inplace=True)
    def calculate_average_fastball (self, pitch_type = 'Cutter'):
        self.radar_df.loc[(self.radar_df['PitchType'] == 'Fastball') & (self.radar_df['AutoPitchType'] != 'Changeup'), 'PitchType'] = self.radar_df['AutoPitchType']
        self.radar_df.loc[(self.radar_df['PitchType'] == 'Fastball') & (self.radar_df['AutoPitchType'] == 'Changeup'), 'PitchType'] = 'Sinker'
        print ("Calculating Fastballs")
        four_seam_averages = self.radar_df[self.radar_df['PitchType'] == pitch_type].groupby('Pitcher').agg({
            'RelSpeed': 'mean',
            'InducedVertBreak': 'mean',
            'HorzBreak': 'mean'
        }).reset_index()

        # Rename columns to reflect they are averages of Four-Seam fastballs
        four_seam_averages.rename(columns={
            'RelSpeed': f'Average{pitch_type}RS',
            'InducedVertBreak': f'Average{pitch_type}IVB',
            'HorzBreak': f'Average{pitch_type}HB'
        }, inplace=True)
        print ('RelSpeed:', f'Average{pitch_type}RS',
               'InducedVertBreak:', f'Average{pitch_type}IVB',
               'HorzBreak:', f'Average{pitch_type}HB')
        # Step 2: Merge the averages back into the original DataFrame
        self.radar_df = pd.merge(self.radar_df, four_seam_averages, on='Pitcher', how='left')
        print (self.radar_df)

    def aggregate_fastball_data (self):
        print ("Aggregating Fastballs")
        self.radar_df['AverageFastballRS_y'] = self.radar_df['AverageFour-SeamRS']
        self.radar_df['AverageFastballIVB_y'] = self.radar_df['AverageFour-SeamIVB']
        self.radar_df['AverageFastballHB_y'] = self.radar_df['AverageFour-SeamHB']
        print ('here')
        self.radar_df['AverageFastballRS_y'] = self.radar_df['AverageFastballRS_y'].fillna(self.radar_df['AverageSinkerRS'])
        self.radar_df['AverageFastballIVB_y'] = self.radar_df['AverageFastballIVB_y'].fillna(self.radar_df['AverageSinkerIVB'])
        self.radar_df['AverageFastballHB_y'] = self.radar_df['AverageFastballHB_y'].fillna(self.radar_df['AverageSinkerHB'])
        self.radar_df['AverageFastballRS_y'] = self.radar_df['AverageFastballRS_y'].fillna(self.radar_df['AverageCutterRS'])
        self.radar_df['AverageFastballIVB_y'] = self.radar_df['AverageFastballIVB_y'].fillna(self.radar_df['AverageCutterIVB'])
        self.radar_df['AverageFastballHB_y'] = self.radar_df['AverageFastballHB_y'].fillna(self.radar_df['AverageCutterHB'])
        self.radar_df['DifferenceRS'] = np.where(self.radar_df['AverageFastballRS_y'].isnull(),
                                                 0,
                                                 self.radar_df['RelSpeed'] - self.radar_df['AverageFastballRS_y'])
        self.radar_df['DifferenceIVB'] = np.where(self.radar_df['AverageFastballIVB_y'].isnull(),
                                                  0,
                                                  self.radar_df['InducedVertBreak'] - self.radar_df['AverageFastballIVB_y'])
        self.radar_df['DifferenceHB'] = np.where(self.radar_df['AverageFastballHB_y'].isnull(),
                                                 0,
                                                 self.radar_df['HorzBreak'] - self.radar_df['AverageFastballHB_y'])
    def write_radar_data (self):
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.radar_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS radar_data')
        with tqdm(total=len(self.radar_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.radar_df), chunk_size):
                end = min(start + chunk_size, len(self.radar_df))
                chunk = self.radar_df.iloc[start:end]
                chunk.to_sql('radar_data', conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def write_variable_data (self, table = 'variables'):#_Pitchers'):
        # numeric_cols = self.input_variables_df.select_dtypes(include='number').columns.tolist()
        #
        # averages_df = self.input_variables_df.groupby(['Pitcher', 'PitchType'])[numeric_cols].mean().reset_index()
        # averages_df[numeric_cols] = averages_df[numeric_cols].apply(lambda x: round(x, 2))
        # averages_df = averages_df.merge(self.input_variables_df[['Pitcher', 'PitcherTeam', 'PitcherThrows']].drop_duplicates(),
        #                                 on='Pitcher', how='left')
        # # Reordering columns to have PitcherTeam and PitcherThrows as the first two columns after Pitcher and PitchType
        # final_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType'] + [col for col in averages_df.columns if col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType']]
        # self.input_variables_df = averages_df[final_columns]
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.input_variables_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        with tqdm(total=len(self.input_variables_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.input_variables_df), chunk_size):
                end = min(start + chunk_size, len(self.input_variables_df))
                chunk = self.input_variables_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def write_predictions_players (self, focus=Focus.Location):
        numeric_cols = self.predictions_df.select_dtypes(include='number').columns.tolist()
        #
        # averages_df = self.predictions_df.groupby(['Pitcher', 'PitchType'])[numeric_cols].mean().reset_index()
        # averages_df[numeric_cols] = averages_df[numeric_cols].apply(lambda x: round(x, 2))
        # averages_df = averages_df.merge(self.predictions_df[['Pitcher', 'PitcherTeam', 'PitcherThrows']].drop_duplicates(),
        #                                 on='Pitcher', how='left')
        # # Reordering columns to have PitcherTeam and PitcherThrows as the first two columns after Pitcher and PitchType
        # final_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType'] + [col for col in averages_df.columns if col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType']]
        # # self.predictions_df = averages_df[final_columns]
        # numeric_cols = self.predictions_df.select_dtypes(include='number').columns.tolist()
        #
        # # Calculate the average of numeric columns grouped by Pitcher and PitchType
        # averages_df = self.predictions_df.groupby(['Pitcher', 'PitchType'])[numeric_cols].mean().reset_index()
        # averages_df[numeric_cols] = averages_df[numeric_cols].apply(lambda x: round(x, 2))
        #
        # # Merge in PitcherTeam and PitcherThrows
        # averages_df = averages_df.merge(
        #     self.predictions_df[['Pitcher', 'PitcherTeam', 'PitcherThrows']].drop_duplicates(),
        #     on='Pitcher', how='left'
        # )
        #
        # # Calculate the 'Usage' column
        # pitch_counts = self.predictions_df.groupby('Pitcher').size().reset_index(name='TotalPitches')
        # pitch_type_counts = self.predictions_df.groupby(['Pitcher', 'PitchType']).size().reset_index(name='PitchTypeCount')
        # usage_df = pitch_type_counts.merge(pitch_counts, on='Pitcher')
        # usage_df['Usage'] = (usage_df['PitchTypeCount'] / usage_df['TotalPitches']) #* 100
        # usage_df['Usage'] = usage_df['Usage'].round(2)
        #
        # # Merge the Usage data back into the averages_df
        # averages_df = averages_df.merge(usage_df[['Pitcher', 'PitchType', 'Usage']], on=['Pitcher', 'PitchType'], how='left')
        #
        # # Reorder columns to put Team and Throws after Pitcher and PitchType
        # final_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage'] + \
        #                 [col for col in averages_df.columns if col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage']]
        # self.predictions_df =self.predictions_df.select_dtypes(include='number').columns.tolist()
        #
        # Calculate the average of numeric columns grouped by Pitcher and PitchType
        #TODO: going to get nuked by name change
        numeric_cols_no_zeros = ['Prob_InPlay', 'xFoul%', 'Prob_SoftGB', 'Prob_HardGB', 'Prob_SoftLD', 'Prob_HardLD', 'Prob_SoftFB', 'Prob_HardFB']  # Specify columns where zeros should be ignored in mean calculation
        agg_dict = {col: 'mean' for col in numeric_cols}  # Default aggregation is mean for all numeric columns
        for col in numeric_cols_no_zeros:
            agg_dict[col] = lambda x: np.mean(x[x != 0]) if len(x[x != 0]) > 0 else np.nan
        averages_df = self.predictions_df.groupby(['Pitcher', 'PitchType']).agg(agg_dict).reset_index()
        # averages_df[numeric_cols] = averages_df[numeric_cols].apply(lambda x: round(x, 2))

        # Merge in PitcherTeam and PitcherThrows
        averages_df = averages_df.merge(
            self.predictions_df[['Pitcher', 'PitcherTeam', 'PitcherThrows']].drop_duplicates(),
            on='Pitcher', how='left'
        )
        # if 'Usage' in averages_df.columns:
        #     averages_df.drop (columns = ['Usage'])
        # else: exit (0)
        # print (averages_df.to_string ())
        # exit (0)
        # Calculate the 'Usage' column
        pitch_counts = self.predictions_df.groupby('Pitcher').size().reset_index(name='TotalPitches')
        pitch_type_counts = self.predictions_df.groupby(['Pitcher', 'PitchType']).size().reset_index(name='PitchTypeCount')
        usage_df = pitch_type_counts.merge(pitch_counts, on='Pitcher')
        usage_df['Usage'] = (usage_df['PitchTypeCount'] / usage_df['TotalPitches']) #* 100
        usage_df['Usage'] = usage_df['Usage'].round(2)
        # Merge the Usage data back into the averages_df
        averages_df = averages_df.merge(usage_df[['Pitcher', 'PitchType', 'Usage']], on=['Pitcher', 'PitchType'], how='left')
        # Reorder columns to put Team and Throws after Pitcher and PitchType
        final_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage'] + \
                        [col for col in averages_df.columns if col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage']]
        predictions_df = averages_df[final_columns]
        table = f'{focus.name}_Probabilities_Pitchers'
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(predictions_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        with tqdm(total=len(predictions_df), desc="Writing to database") as pbar:
            for start in range(0, len(predictions_df), chunk_size):
                end = min(start + chunk_size, len(predictions_df))
                chunk = predictions_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()
    #TODO: this, um, does not work, but I thought about it and I'm not actually sure it's necessary
    def find_overall_percentiles (self):
        conn = sqlite3.connect(f'{self.db_file}')
        table_name = 'Location_Probabilities_Pitchers'
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close ()
        # df['New Prob'] = df['Usage'] * df['Prob']
        # df = df.groupby('Pitcher')['New Prob'].sum().reset_index()
        df.drop (columns = ['PitcherTeam'])
        for col in df.columns:
            if col != 'Pitcher' and col != 'PitchType':
                # print (col)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df_long = pd.melt(df, id_vars=['Pitcher', 'PitchType', 'Usage'], var_name='ProbType', value_name='Prob')

        # Calculate 'New Prob' based on 'Usage' and 'Prob'
        df_long['New Prob'] = df_long['Usage'] * df_long['Prob']

        # Group by Pitcher and ProbType, then aggregate 'New Prob' by summing
        df = df_long.groupby(['Pitcher', 'ProbType'])['New Prob'].sum().reset_index()

        self.write_df(df, 'Location_Probabilities_Pitchers_No_Pitch_Type')
    def write_predictions (self, focus=Focus.Location):
        table = f'{focus.name}_Probabilities'
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.predictions_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        with tqdm(total=len(self.predictions_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.predictions_df), chunk_size):
                end = min(start + chunk_size, len(self.predictions_df))
                chunk = self.predictions_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def write_df (self, df, table):
        def write_predictions (self, focus=Focus.Location):
            table = f'{focus.name}_Probabilities'
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        with tqdm(total=len(df), desc="Writing to database") as pbar:
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk = df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()


    def write_players (self, focus=Focus.Location):
        table = f'Pitcher_{focus.name}_Ratings_20_80_scale'
        # table = f'Pitcher_{focus.name}_Ratings_100_scale'
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.players_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        with tqdm(total=len(self.players_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.players_df), chunk_size):
                end = min(start + chunk_size, len(self.players_df))
                chunk = self.players_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def write_percentiles (self, focus=Focus.Location):
        final_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage'] + \
                        [col for col in self.percentiles_df.columns if col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchType', 'Usage', 'EV', 'xRV', 'average_xRV']] + \
                        ['xRV']
        self.percentiles_df = self.percentiles_df[final_columns]
        table = f'Percentiles_{focus.name}_Pitchers'
        # table = f'Pitcher_{focus.name}_Ratings_100_scale'
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.percentiles_df) // chunk_size + 1
        # print (self.percentiles_df)
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS {table}')
        conn.commit()
        with tqdm(total=len(self.percentiles_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.percentiles_df), chunk_size):
                end = min(start + chunk_size, len(self.percentiles_df))
                chunk = self.percentiles_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def write_current_data (self, table = 'variables'):
        chunk_size = 1000  # Adjust based on your needs and system capabilities
        num_chunks = len(self.current_df) // chunk_size + 1
        conn = sqlite3.connect(f'{self.db_file}')
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        with tqdm(total=len(self.current_df), desc="Writing to database") as pbar:
            for start in range(0, len(self.current_df), chunk_size):
                end = min(start + chunk_size, len(self.current_df))
                chunk = self.current_df.iloc[start:end]
                chunk.to_sql(table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))
        conn.close()

    def remove_column (self, column):
        self.radar_df = self.radar_df.drop(column, axis=1)

    def rename_column(self, old_column_name, new_column_name):
        self.radar_df.rename(columns={old_column_name: new_column_name}, inplace=True)

    def table_to_excel (self, table_name):
        conn = sqlite3.connect(self.db_file)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE PitcherTeam = 'VIR_CAV'", conn)
        # df = pd.read_sql_query("SELECT * FROM {table_name}", conn)
        conn.close()
        df.to_excel(f'UVA_{table_name}.xlsx', index=False)

    def load_relevant_data (self):
        selected_columns = ['PitchUID',
                            'Date',
                            'PitcherTeam',
                            'BatterTeam',
                            'PitchType',
                            'PitchCall',
                            'TaggedHitType',
                            'ExitSpeed',
                            'Pitcher',
                            'PitcherThrows',
                            'Batter',
                            'BatterSide',
                            'Balls',
                            'Strikes',

                            'RelSpeed',
                            'ZoneSpeed',
                            'InducedVertBreak',
                            'HorzBreak',
                            'SpinRate',
                            'SpinEfficiency',
                            'AxisDifference',
                            'VAA',
                            'RelHeight',
                            'RelSide',
                            'Extension',
                            'VertRelAngle',
                            'HorzRelAngle',
                            'DifferenceRS',
                            'DifferenceIVB',
                            'DifferenceHB',

                            'PlateLocHeight',
                            'PlateLocSide']
        self.input_variables_df = self.radar_df[selected_columns].copy()

    def clean_data_for_take_model (self, training = 1):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'Ball') |
            (self.input_variables_df['PitchCall'] == 'StrikeCalled') |
            (self.input_variables_df['PitchCall'] == 'HitByPitch')
            ]
        if (training == 0):
            self.current_df = self.input_variables_df
        value_map = {
            'HitByPitch' : 2,
            'Ball' : 1,
            'StrikeCalled' : 0
        }
        self.current_df['Target'] = self.current_df['PitchCall'].replace(value_map)
        self.param = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
        }
        self.multi = True
        self.currently_modeling = 'Take'

    def clean_data_for_swing_model (self, training = 1):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'Ball') |
            (self.input_variables_df['PitchCall'] == 'StrikeCalled') |
            (self.input_variables_df['PitchCall'] == 'HitByPitch') |
            (self.input_variables_df['PitchCall'] == 'InPlay') |
            (self.input_variables_df['PitchCall'] == 'StrikeSwinging') |
            (self.input_variables_df['PitchCall'] == 'Foul')
            ]
        if (training == 0):
            self.current_df = self.input_variables_df
        value_map = {
            'Ball' : 0,
            'StrikeCalled' : 0,
            'HitByPitch' : 0,
            'InPlay' : 1,
            'StrikeSwinging' : 1,
            'Foul' : 1
        }
        self.current_df['Target'] = self.current_df['PitchCall'].replace(value_map)
        self.param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        self.multi = False
        self.currently_modeling = 'Swing'
    def clean_data_for_contact_model (self, training = 1):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') |
            (self.input_variables_df['PitchCall'] == 'StrikeSwinging') |
            (self.input_variables_df['PitchCall'] == 'Foul')
            ]
        if (training == 0):
            self.current_df = self.input_variables_df
        value_map = {
            'InPlay' : 1,
            'StrikeSwinging' : 0,
            'Foul' : 1
        }
        self.current_df['Target'] = self.current_df['PitchCall'].replace(value_map)
        self.param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        self.multi = False
        self.currently_modeling = 'Contact'

    def clean_data_for_foul_model (self, training = 1):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') |
            (self.input_variables_df['PitchCall'] == 'Foul')
            ]
        if (training == 0):
            self.current_df = self.input_variables_df
        value_map = {
            'InPlay' : 0,
            'Foul' : 1
        }
        self.current_df['Target'] = self.current_df['PitchCall'].replace(value_map)
        self.param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        self.multi = False
        self.currently_modeling = 'Foul'

    def clean_data_for_in_play_model (self, training = 1):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') &
            (self.input_variables_df['ExitSpeed'].notna()) &
            (self.input_variables_df['TaggedHitType'] != 'Undefined')
            ]
        if (training == 0):
            self.current_df = self.input_variables_df
        # print (self.input_variables_df)
        # self.input_variables_df = self.input_variables_df ['ExitSpeed'].dropna()
        # print (self.input_variables_df)
        def map_exit_speed(row):
            speed = row['ExitSpeed']
            hit_type = row['TaggedHitType']
            base_map = {'GroundBall': 0, 'LineDrive': 2, 'FlyBall': 4}
            offset = base_map[hit_type]

            if speed < 90:
                return offset
            elif 90 <= speed < 95:
                return offset
            elif 95 <= speed < 100:
                return offset + 1
            elif 100 <= speed < 105:
                return offset + 1
            else:  # 105+
                return offset + 1

        # Apply the function to each row
        if (training == 1):
            self.current_df['Target'] = self.current_df.apply(map_exit_speed, axis=1)
        # self.input_variables_df['Target'] = self.input_variables_df.apply(assign_target, axis=1, breakpoints=breakpoints, gb_labels=gb_labels, ld_labels=ld_labels, fb_labels=fb_labels)
        # self.input_variables_df = pd.concat([self.input_variables_df, df_fb])
        # self.input_variables_df['Target'] = self.input_variables_df['PitchCall'].replace(value_map)
        self.param = {
            'objective': 'multi:softprob',
            'num_class': 6,
            # 'eval_metric': 'mlogloss',
            'eval_metric': 'mlogloss',
        }
        # self.param = {
        #     'objective': 'binary:logistic',
        #     'eval_metric': 'logloss'
        # }
        self.multi = True
        self.currently_modeling = 'InPlay'

    # def clean_data_for_grounder_model (self):
    #     self.current_df = self.input_variables_df[
    #         (self.input_variables_df['PitchCall'] == 'InPlay') &
    #         (self.input_variables_df['ExitSpeed'].notna()) &
    #         (self.input_variables_df['TaggedHitType'] != 'Undefined')
    #         ]
    #     value_map = {
    #         'GroundBall' : 1,
    #         'FlyBall' : 0,
    #         'LineDrive' : 0
    #     }
    #     self.multi = False
    #     self.current_df['Target'] = self.current_df['TaggedHitType'].replace(value_map)
    #     self.param = {
    #         'objective': 'binary:logistic',
    #         'eval_metric': 'logloss'
    #     }
    #     self.currently_modeling = 'Grounder'

    def clean_data_for_hit_type_model (self):
        self.current_df = self.input_variables_df[
            (self.input_variables_df['PitchCall'] == 'InPlay') &
            (self.input_variables_df['ExitSpeed'].notna()) &
            (self.input_variables_df['TaggedHitType'] != 'Undefined')
            ]
        value_map = {
            'FlyBall' : 2,
            'LineDrive' : 1,
            'GroundBall' : 0
        }
        self.param = {
            'objective': 'multi:softprob',
            'num_class': 3,
            # 'eval_metric': 'mlogloss',
            'eval_metric': 'merror',
        }
        self.multi = True
        self.current_df['Target'] = self.current_df['TaggedHitType'].replace(value_map)
        # self.param = {
        #     'objective': 'binary:logistic',
        #     'eval_metric': 'logloss'
        # }
        self.currently_modeling = 'FlyBall'

    # def clean_data_for_linedrive_model (self):
    #     self.current_df = self.input_variables_df[
    #         (self.input_variables_df['PitchCall'] == 'InPlay') &
    #         (self.input_variables_df['ExitSpeed'].notna()) &
    #         (self.input_variables_df['TaggedHitType'] != 'Undefined')
    #         ]
    #     value_map = {
    #         'FlyBall' : 0,
    #         'LineDrive' : 1,
    #         'GroundBall' : 0
    #     }
    #     self.current_df['Target'] = self.current_df['TaggedHitType'].replace(value_map)
    #     self.param = {
    #         'objective': 'binary:logistic',
    #         'eval_metric': 'logloss'
    #     }
    #     self.multi = False
    #     self.currently_modeling = 'LineDrive'
    #
    def clean_data_for_fastballs (self):
        self.features = [
            'PitchType', 'PitcherThrows', 'BatterSide'
        ]
        # self.features.extend (self.context_features)
        # print (self.current_df.to_string ())
        # exit (0)
        self.current_df = self.current_df[
            (self.current_df['PitchType'] == 'Four-Seam') |
            (self.current_df['PitchType'] == 'Sinker') #|
            # (self.current_df['PitchType'] == 'Cutter')
            ]
        # self.input_variables_df = self.input_variables_df.drop('DifferenceRS', axis=1)
        # self.input_variables_df = self.input_variables_df.drop('DifferenceIVB', axis=1)
        # self.input_variables_df = self.input_variables_df.drop('DifferenceHB', axis=1)
        self.current_pitch_class = 'Fastball'
        # print (self.input_variables_df)
    #     # self.remove_column ('AverageFastballRS')
    #     # self.remove_column ('AverageFastballIVB')
    #     # self.remove_column ('AverageFastballHB')

    def clean_data_for_breakingballs (self):
        self.features = [
            'PitchType', 'PitcherThrows', 'BatterSide'
        ]
        # self.features.extend (self.context_features)
        self.current_df = self.current_df[
            (self.current_df['PitchType'] == 'Slider') |
            (self.current_df['PitchType'] == 'Curveball') |
            (self.current_df['PitchType'] == 'Cutter')#_S')
            ]
        self.current_pitch_class = 'BreakingBall'

    def clean_data_for_offspeed (self):
        self.features = [
            'PitchType', 'PitcherThrows', 'BatterSide'
        ]
        # self.features.extend (self.context_features)
        self.current_df = self.current_df[
            (self.current_df['PitchType'] == 'ChangeUp') |
            (self.current_df['PitchType'] == 'Splitter')
            ]
        self.current_pitch_class = 'Offspeed'

    def train_classifier (self):
        features = self.features + self.context_features
        X = self.current_df[features]
        y = self.current_df['Target']
        # if (self.multi):
        #     y = self.current_df['Target1', 'Target2', 'Target3']
        # print (self.input_variables_df.to_string ())
        # print (X)
        X_encoded = pd.get_dummies(X, columns=['PitchType', 'PitcherThrows', 'BatterSide'])
        X['PitchType'] = X['PitchType'].astype('category')
        X['PitcherThrows'] = X['PitcherThrows'].astype('category')
        X['BatterSide'] = X['BatterSide'].astype('category')
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
                'early_stopping_rounds': stopping_rounds
                # 'objective': 'multi:softprob',
                # 'num_class': 15,
                # 'eval_metric': 'mlogloss',
                # 'enable_categorical': True
            }

            clf = xgb.XGBClassifier(**param,**self.param, enable_categorical=True)
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            if (self.multi):
                preds = clf.predict_proba(X_test)
            else:
                preds = clf.predict_proba(X_test)[:, 1]
            trial.set_user_attr("model", clf)
            binary_preds = np.where(preds > 0.5, 1, 0)
            # score = f1_score(y_test, binary_preds)
            # score = roc_auc_score(y_test, preds)
            # class_preds = np.argmax(preds, axis=1)
            score = log_loss(y_test, preds)
            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=85)
        best_model = study.best_trial.user_attrs["model"]
        best_params = study.best_trial.params
        model_directory = "Full_Model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Save the best model to a file within the 'Model' directory
        model_filename = os.path.join(model_directory, f'joblib_model_{self.focus.name}_{self.currently_modeling}--{self.current_pitch_class}.json')
        # model_filename = f'{self.focus.name}_{self.currently_modeling}-{self.current_pitch_class}.json' # or use .bin for binary format
        best_model.save_model(model_filename)

        model_directory = "JobLib_Model_Location"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename = os.path.join(model_directory, f'joblib_model_{self.focus.name}_{self.currently_modeling}--{self.current_pitch_class}.joblib')
        # Save the best model to a file
        joblib.dump(best_model, model_filename)
        # Optionally, save the best parameters to a file as well
        with open(f'best_params_{self.focus.name}_{self.currently_modeling}.txt', 'w') as f:
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
        self.current_df['PitchType'] = self.current_df['PitchType'].astype('category')
        self.current_df['PitcherThrows'] = self.current_df['PitcherThrows'].astype('category')
        self.current_df['BatterSide'] = self.current_df['BatterSide'].astype('category')
        # self.input_variables_df['PredictedClass'] = clf.predict(self.input_variables_df[features])
        # self.input_variables_df['PredictedProbability'] = clf.predict_proba(self.input_variables_df[features])[:, 1]
        class_labels = clf.classes_
        probabilities = clf.predict_proba(self.current_df[features])
        # Create a new column for each class probability
        for i, class_label in enumerate(class_labels):
            self.current_df[f'Prob_{class_label}'] = probabilities[:, i]
        self.write_current_data(f'{self.focus.name}_{self.currently_modeling}-{self.current_pitch_class}')

    #post prediction flowchart
    # predictions -> Location_Probabilities
    # -> Location_Probabilities_Pitchers
    # -> Pitcher_Location_Ratings_20_80
    # -> Percentiles_Location_Pitchers
    def load_predictions (self):
        predictions_df = self.input_variables_df
        # new_columns = ['Prob_SwingingStrike', 'Prob_Contact', 'Prob_InPlay', 'Prob_Foul', 'Prob_SoftGB', 'Prob_HardGB', 'Prob_SoftLD', 'Prob_HardLD','Prob_SoftFB', 'Prob_HardFB']
        #
        # Adding new empty columns
        # for column in new_columns:
        #     predictions_df[column] = np.nan

        conn = sqlite3.connect(f'{self.db_file}')
        TBB_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2" FROM "Location_Take-BreakingBall"', conn)
        TBB_df = TBB_df.rename (columns = {"Prob_0" : "Prob_CS", "Prob_1" : "Prob_Ball", "Prob_2" : "Prob_HBP"})
        TF_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2" FROM "Location_Take-Fastball"', conn)
        TF_df = TF_df.rename (columns = {"Prob_0" : "Prob_CS", "Prob_1" : "Prob_Ball", "Prob_2" : "Prob_HBP"})
        TO_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2" FROM "Location_Take-Offspeed"', conn)
        TO_df = TO_df.rename (columns = {"Prob_0" : "Prob_CS", "Prob_1" : "Prob_Ball", "Prob_2" : "Prob_HBP"})
        take_df = pd.concat([TF_df, TBB_df, TO_df], axis=0)
        take_df.reset_index(drop=True, inplace=True)

        SBB_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Swing-BreakingBall"', conn)
        SBB_df = SBB_df.rename (columns = {"Prob_0" : "xTake%", "Prob_1" : "xSwing%"})
        SF_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Swing-Fastball"', conn)
        SF_df = SF_df.rename (columns = {"Prob_0" : "xTake%", "Prob_1" : "xSwing%"})
        SO_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Swing-Offspeed"', conn)
        SO_df = SO_df.rename (columns = {"Prob_0" : "xTake%", "Prob_1" : "xSwing%"})
        swing_df = pd.concat([SF_df, SBB_df, SO_df], axis=0)
        swing_df.reset_index(drop=True, inplace=True)

        CBB_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Contact-BreakingBall"', conn)
        CBB_df = CBB_df.rename (columns = {"Prob_0" : "xWhiff%", "Prob_1" : "Prob_Contact"})
        CF_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Contact-Fastball"', conn)
        CF_df = CF_df.rename (columns = {"Prob_0" : "xWhiff%", "Prob_1" : "Prob_Contact"})
        CO_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Contact-Offspeed"', conn)
        CO_df = CO_df.rename (columns = {"Prob_0" : "xWhiff%", "Prob_1" : "Prob_Contact"})
        con_df = pd.concat([CF_df, CBB_df, CO_df], axis=0)
        con_df.reset_index(drop=True, inplace=True)

        FBB_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Foul-BreakingBall"', conn)
        FBB_df = FBB_df.rename (columns = {"Prob_0" : "Prob_InPlay", "Prob_1" : "xFoul%"})
        FF_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Foul-Fastball"', conn)
        FF_df = FF_df.rename (columns = {"Prob_0" : "Prob_InPlay", "Prob_1" : "xFoul%"})
        FO_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1" FROM "Location_Foul-Offspeed"', conn)
        FO_df = FO_df.rename (columns = {"Prob_0" : "Prob_InPlay", "Prob_1" : "xFoul%"})
        foul_df = pd.concat([FF_df, FBB_df, FO_df], axis=0)
        foul_df.reset_index(drop=True, inplace=True)

        IBB_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2", "Prob_3", "Prob_4", "Prob_5", "Target" FROM "Location_InPlay-BreakingBall"', conn)
        IBB_df = IBB_df.rename (columns = {"Prob_0" : "Prob_SoftGB", "Prob_1" : "Prob_HardGB", "Prob_2" : "Prob_SoftLD", "Prob_3" : "Prob_HardLD", "Prob_4" : "Prob_SoftFB", "Prob_5" : "Prob_HardFB"})
        IF_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2", "Prob_3", "Prob_4", "Prob_5", "Target" FROM "Location_InPlay-Fastball"', conn)
        IF_df = IF_df.rename (columns = {"Prob_0" : "Prob_SoftGB", "Prob_1" : "Prob_HardGB", "Prob_2" : "Prob_SoftLD", "Prob_3" : "Prob_HardLD", "Prob_4" : "Prob_SoftFB", "Prob_5" : "Prob_HardFB"})
        IO_df = pd.read_sql_query('SELECT "PitchUID", "Prob_0", "Prob_1", "Prob_2", "Prob_3", "Prob_4", "Prob_5", "Target" FROM "Location_InPlay-Offspeed"', conn)
        IO_df = IO_df.rename (columns = {"Prob_0" : "Prob_SoftGB", "Prob_1" : "Prob_HardGB", "Prob_2" : "Prob_SoftLD", "Prob_3" : "Prob_HardLD", "Prob_4" : "Prob_SoftFB", "Prob_5" : "Prob_HardFB"})
        inplay_df = pd.concat([IF_df, IBB_df, IO_df], axis=0)
        inplay_df.reset_index(drop=True, inplace=True)

        predictions_df = predictions_df.merge (take_df, on='PitchUID', how='left')
        predictions_df = predictions_df.merge (swing_df, on='PitchUID', how='left')
        predictions_df = predictions_df.merge (con_df, on='PitchUID', how='left')
        predictions_df = predictions_df.merge (foul_df, on='PitchUID', how='left')
        predictions_df = predictions_df.merge (inplay_df, on='PitchUID', how='left')
        # predictions_df = predictions_df[
        #     (predictions_df['PitchCall'] == 'InPlay') |
        #     (predictions_df['PitchCall'] == 'StrikeSwinging') |
        #     (predictions_df['PitchCall'] == 'Foul')
        #     ]
        predictions_df.to_sql (f'{self.focus.name}_Probabilities', conn, if_exists='replace', index=False)
        self.predictions_df = predictions_df
        conn.close ()

    def calculate_run_values_swing (self):
        strike_values = {
            (0, 0): -0.048,#0.037
            (0, 1): -0.066,#0.051
            (0, 2): -0.196,#0.151
            (1, 0): -0.062,#0.048
            (1, 1): -0.070,#0.054
            (1, 2): -0.223,#0.172
            (2, 0): -0.084,#0.065
            (2, 1): -0.079,#0.061
            (2, 2): -0.273,#0.210
            (3, 0): -0.152,#0.117
            (3, 1): -0.086,#0.066
            (3, 2): -0.384,#0.295
        }
        foul_values = {
            (0, 0): -0.048,#0.037
            (0, 1): -0.066,#0.051
            (0, 2): 0,#0.151
            (1, 0): -0.062,#0.048
            (1, 1): -0.070,#0.054
            (1, 2): 0,#0.172
            (2, 0): -0.084,#0.065
            (2, 1): -0.079,#0.061
            (2, 2): 0,#0.210
            (3, 0): -0.152,#0.117
            (3, 1): -0.086,#0.066
            (3, 2): 0,#0.295
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
        woba_scale = 1.3
        expected_run_values = {
            "HBP" : -0.3,
            "SoftGB": 0,
            "HardGB": -0.1,
            "SoftLD": -0.25,
            "HardLD": -0.75,
            "SoftFB": 0,
            "HardFB": -1
        }

        predictions_df = self.predictions_df
        predictions_df = predictions_df.dropna(subset=['xWhiff%'])
        predictions_df = predictions_df.fillna(0)
        def calculate_strike_value(row):
            count = (row['Balls'], row['Strikes'])
            swinging_strike_value = strike_values.get(count, 0)
            ev = swinging_strike_value
            return ev
        def calculate_ball_value(row):
            count = (row['Balls'], row['Strikes'])
            ball_value = ball_values.get(count, 0)
            ev = ball_value
            return ev
        def calculate_foul_value(row):
            count = (row['Balls'], row['Strikes'])
            foul_value = foul_values.get(count, 0)
            ev = foul_value
            return ev

        # Apply the calculate_ev function row-wise
        predictions_df['StrikeValue'] = predictions_df.apply(calculate_strike_value, axis=1)
        predictions_df['BallValue'] = predictions_df.apply(calculate_ball_value, axis=1)
        predictions_df['FoulValue'] = predictions_df.apply(calculate_foul_value, axis=1)
        predictions_df['EV'] = (
                predictions_df['xTake%'] * predictions_df ['Prob_CS'] * predictions_df["StrikeValue"]
                + predictions_df['xTake%'] * predictions_df ['Prob_Ball'] * predictions_df["BallValue"]
                + predictions_df['xTake%'] * predictions_df ['Prob_HBP'] * expected_run_values["HBP"]
                + predictions_df['xSwing%'] * predictions_df['xWhiff%'] * predictions_df["StrikeValue"]
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['xFoul%'] * predictions_df['FoulValue']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_SoftGB'] * expected_run_values['SoftGB']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_HardGB'] * expected_run_values['HardGB']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_SoftLD'] * expected_run_values['SoftLD']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_HardLD'] * expected_run_values['HardLD']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_SoftFB'] * expected_run_values['SoftFB']
                + predictions_df['xSwing%'] * predictions_df['Prob_Contact'] * predictions_df['Prob_InPlay'] * predictions_df['Prob_HardFB'] * expected_run_values['HardFB']
        )
        ev = predictions_df ['EV'].mean ()
        predictions_df ['xRV'] = predictions_df ['EV'] - ev
        predictions_df['average_xRV'] = predictions_df.groupby(['Pitcher', 'PitchType'])['xRV'].transform('mean')
        self.predictions_df = predictions_df

    def calculate_average_xRVs (self):
        predictions_df = self.predictions_df
        # predictions_df['average_xRV'] = predictions_df.groupby(['Pitcher', 'PitchType'])['xRV'].transform('mean')
        # pitch_stats_df = predictions_df.groupby('PitchType')['xRV'].agg(['mean', 'std']).reset_index()
        # pitch_stats_df.columns = ['PitchType', 'Average_xRV', 'StDev_xRV']
        #
        # # Calculate overall average and standard deviation of xRV across all pitch types
        # overall_stats = predictions_df['xRV'].agg(['mean', 'std']).to_frame().T
        # overall_stats.insert(0, 'PitchType', 'Overall')
        # overall_stats.columns = ['PitchType', 'Average_xRV', 'StDev_xRV']
        pitcher_total_pitches = predictions_df.groupby('Pitcher')['PitchType'].transform('count')
        pitch_type_count = predictions_df.groupby(['Pitcher', 'PitchType'])['PitchType'].transform('count')
        predictions_df['Usage'] = (pitch_type_count / pitcher_total_pitches).round(2)
        # print (predictions_df)
        predictions_df = predictions_df [predictions_df ['Usage'] >= 0.04]
        # predictions_df = predictions_df[pitcher_total_pitches >= 20]
        predictions_df = predictions_df.drop_duplicates(subset=['PitchUID'], keep='first')
        predictions_df.drop ('Usage', axis=1)
        # print (predictions_df)
        # exit (0)
        temp_df = predictions_df.copy ()
        temp_df = temp_df[pitcher_total_pitches >= 50]
        temp_df['Overall_average_xRV'] = predictions_df.groupby('Pitcher')['xRV'].transform('mean')
        temp_df = temp_df.drop_duplicates(subset=['Pitcher'], keep='first')
        pitch_stats_df = temp_df.groupby('PitchType')['average_xRV'].agg(['mean', 'std']).reset_index()
        pitch_stats_df.columns = ['PitchType', 'Average_xRV', 'StDev_xRV']

        # Calculate overall average and standard deviation of the average xRV across all pitch types
        overall_stats = temp_df['Overall_average_xRV'].agg(['mean', 'std']).to_frame().T
        overall_stats.insert(0, 'PitchType', 'Overall')
        overall_stats.columns = ['PitchType', 'Average_xRV', 'StDev_xRV']


        xRV_df = pd.concat([pitch_stats_df, overall_stats], ignore_index=True)
        self.xRV_df = xRV_df
        print (xRV_df.to_string ())
        avg_xRV = predictions_df.groupby(['Pitcher', 'PitchType'])['average_xRV'].mean().reset_index()
        # avg_xRV = avg_xRV [avg_xRV ['Pitcher'] == 'Hungate, Chase']
        # print (avg_xRV.to_string ())
        # exit (0)
        # Step 2: Calculate Z-Scores for each Pitcher and Pitch Type
        # Merge the average xRV with the mean and std from xRV_df
        avg_xRV = avg_xRV.merge(xRV_df[['PitchType', 'Average_xRV', 'StDev_xRV']], on='PitchType', how='left')

        # Calculate the z-score
        avg_xRV['z_score'] = round ((avg_xRV['average_xRV'] - avg_xRV['Average_xRV']) / avg_xRV['StDev_xRV'] * 10 + 50, 2)
        # avg_xRV = avg_xRV [avg_xRV ['Pitcher'] == 'Hungate, Chase']
        # print (avg_xRV.to_string ())
        # exit (0)
        # avg_xRV['z_score'] = round (avg_xRV['average_xRV']/avg_xRV['Average_xRV'] * 100, 2)
        players_df = avg_xRV.pivot(index='Pitcher', columns='PitchType', values='z_score').reset_index()
        overall_avg_xRV = predictions_df.groupby('Pitcher')['xRV'].mean ().reset_index()
        # print (overall_avg_xRV)
        # exit (0)
        overall_stats = xRV_df[xRV_df['PitchType'] == 'Overall'][['Average_xRV', 'StDev_xRV']].iloc[0]
        overall_avg_xRV['Overall_z_score'] = round ((overall_avg_xRV['xRV'] - overall_stats['Average_xRV']) / overall_stats['StDev_xRV'] * 10 + 50, 2)
        # overall_avg_xRV['Overall_z_score'] = round (overall_avg_xRV['xRV']/avg_xRV['Average_xRV'] * 100, 2)
        # print (overall_avg_xRV)
        # exit (0)
        # Merge this overall z-score back into players_df
        players_df = players_df.merge(overall_avg_xRV[['Pitcher', 'Overall_z_score']], on='Pitcher', how='left')
        pitch_counts = predictions_df.groupby('Pitcher')['PitchUID'].nunique().reset_index(name='PitchCount')
        players_df = players_df.merge(pitch_counts, on='Pitcher', how='left')
        # print (players_df)
        # exit (0)


        # Rename columns appropriately and handle NaNs for pitchers who do not use certain pitch types
        players_df = players_df.rename(columns={'Overall_z_score': 'Overall'}).fillna(np.nan)
        pitcher_team_mapping = predictions_df[['Pitcher', 'PitcherTeam', 'PitcherThrows']].drop_duplicates()
        players_df = players_df.merge(pitcher_team_mapping, on='Pitcher', how='left')

        pitch_counts = predictions_df.groupby('Pitcher').size().reset_index(name='TotalPitches')
        pitch_type_counts = predictions_df.groupby(['Pitcher', 'PitchType']).size().reset_index(name='PitchTypeCount')
        usage_df = pitch_type_counts.merge(pitch_counts, on='Pitcher')
        usage_df['Usage'] = (usage_df['PitchTypeCount'] / usage_df['TotalPitches']).round(2)
        # usage_df = usage_df.replace(0, np.nan)
        usage_2d_df = usage_df.pivot(index='Pitcher', columns='PitchType', values='Usage')
        usage_2d_df.columns = [f"{col} Usage" for col in usage_2d_df.columns]
        usage_2d_df_reset = usage_2d_df.reset_index()
        players_df = players_df.merge(usage_2d_df_reset, on='Pitcher', how='left')
        base_columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchCount', 'Overall']#, 'Four-Seam', 'Four-Seam Usage', 'Sinker', 'Sinker Usage', 'Cutter', 'Cutter Usage', 'Cutter_S', 'Cutter_S Usage', 'Curveball', 'Curveball Usage', 'Slider', 'Slider Usage', 'ChangeUp', 'ChangeUp Usage', 'Splitter', 'Splitter Usage']
        usage_columns1 = [col for col in players_df.columns if (not col.endswith('Usage')) and (col not in ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchCount', 'Overall'])]
        usage_columns2 = [col for col in players_df.columns if col.endswith('Usage')]
        final_columns = base_columns + usage_columns1 + usage_columns2
        players_df = players_df[final_columns]
        # exit (0)
        self.players_df = players_df

    def calculate_percentiles (self, focus=Focus.Location):
        print ("Reading player predictions")
        table_name = f'{focus.name}_Probabilities_Pitchers'
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {table_name}', conn).iloc[0, 0]
        chunksize = 10000
        pbar = tqdm(total=total_rows)
        df_list = []
        for chunk in pd.read_sql_query(f'SELECT * FROM {table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])
        players_df = pd.concat(df_list, ignore_index=True)
        # print (players_df)
        pbar.close()
        conn.close()
        # def percentile_within_group(x):
        #     return x.rank(pct=True)
        # players_df = players_df
        # # Group by 'PitchType' and apply the percentile calculation to each numerical column
        # percentiles_df = players_df.groupby('PitchType').transform(percentile_within_group)
        #
        # # Combine with non-numerical columns
        # result_df = pd.concat([players_df[['Pitcher', 'PitchType']], percentiles_df], axis=1)
        # self.percentiles_df = result_df
        #TODO old names
        players_df ['xGB%'] = players_df ['Prob_SoftGB'] + players_df ['Prob_HardGB']
        players_df ['xHH%'] = players_df ['Prob_HardGB'] + players_df ['Prob_HardLD'] + players_df ['Prob_HardFB']
        players_df ['DifferenceRS'] = abs (players_df ['DifferenceRS'])
        players_df ['DifferenceIVB'] = abs (players_df ['DifferenceIVB'])
        players_df ['DifferenceHB'] = abs (players_df ['DifferenceHB'])
        # print (players_df)
        # print (predictions_df)
        players_df = players_df [players_df ['Usage'] >= 0.04]
        #players_df = players_df.drop_duplicates(subset=['Pitcher'], keep='first')
        players_df = players_df.drop_duplicates(subset=['Pitcher', 'PitchType'])

        #Turn to NaN instead of drop
        # duplicates = df.duplicated(subset=['Pitcher', 'PitchType'], keep='first')
        #
        # # Replace non-Pitcher and non-PitcherTeam columns with NaN for duplicates
        # columns_to_nan = df.columns.difference(['Pitcher', 'PitcherTeam'])
        # df.loc[duplicates, columns_to_nan] = np.nan


        # print (players_df)
        # exit (0)
        def calculate_percentiles(df, group_field):
            # Group the DataFrame by PitchType and then apply the percentile function
            percentiles_df = df.groupby(group_field).transform(lambda x: round(100*x.rank(pct=True),0) if np.issubdtype(x.dtype, np.number) else x)
            # Keep non-numeric columns as they are
            for col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    percentiles_df[col] = df[col]
                elif col == 'Usage':
                    percentiles_df[col] = df[col]
            return percentiles_df

        players_percentiles_df = calculate_percentiles(players_df, 'PitchType')
        # print (players_percentiles_df)
        self.percentiles_df = players_percentiles_df
        # print (percentiles_df.to_string ())

    def createPlots (self, x_axis = 'HorzBreak', y_axis = 'InducedVertBreak', heat = 'xRV', pitch_type = 'Four-Seam'):
        data = {
            'horizontal_break': np.random.randn(1000),
            'vertical_break': np.random.randn(1000),
            'Location_plus': np.random.rand(1000) * 200  # Location+ values from 0 to 200
        }
        df = pd.DataFrame(data)
        # Modify the colormap to center the yellow color at 0 Location+

        # Now let's modify the hexbin plot to use the data from the DataFrame
        plt.figure(figsize=(10, 8))
        pitch_type = 'Sinker'
        self.predictions_df ['xGB%'] = self.predictions_df ['Prob_SoftGB'] + self.predictions_df ['Prob_HardGB']
        self.predictions_df ['xHH%'] = self.predictions_df ['Prob_HardGB'] + self.predictions_df ['Prob_HardLD'] + self.predictions_df ['Prob_HardFB']
        self.predictions_df = self.predictions_df [self.predictions_df ['PitchType'] == pitch_type]
        self.predictions_df = self.predictions_df [self.predictions_df ['xHH%'] > 0]
        vmin = 0#-.1#self.predictions_df ['xRV'].min()
        vmax = .8#0.1
        vcenter = 0.4
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        x_axis = 'AxisDifference'
        y_axis = 'SpinEfficiency'
        heat = 'xHH%'
        cmap_colors = ["#1a1cf4", "#ffdd7d", "#ff191a"]
        custom_colormap = LinearSegmentedColormap.from_list("custom_color_map", cmap_colors)

        hb = plt.hexbin(self.predictions_df [x_axis], self.predictions_df[y_axis], C=self.predictions_df[heat],
                        gridsize=30, cmap=custom_colormap, reduce_C_function=np.mean, norm=norm, extent = [-30, 30, 0, 1])

        # Add a color bar
        cb = plt.colorbar(hb, spacing='proportional', label=heat)
        cb.set_label(heat)
        # cmap_colors = [(0.0, 0x2922c2), (0.5, 0xffdd7d), (1, 0xda3127)]
        # Use this colormap in a plot
        # plt.imshow([[0,1]], cmap=custom_colormap)

        plt.axvline(self.predictions_df [x_axis].mean (), linestyle = 'dashed', color='k', linewidth=2)
        plt.axhline(self.predictions_df [y_axis].mean (), linestyle = 'dashed', color='k', linewidth=2)

        # Add labels and title
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(pitch_type)

        # Show the plot
        plt.show()
        plt.close ()
        density = gaussian_kde(self.predictions_df['AxisDifference'])
        density.covariance_factor = lambda: .25  # Adjust smoothing here, lower values give more detail
        density._compute_covariance()

        # Set up the x values over which you want to evaluate the density function
        x = np.linspace(min(self.predictions_df['AxisDifference']), max(self.predictions_df['AxisDifference']), 1000)

        # Plot the density
        plt.plot(x, density(x))

        plt.title('Line Histogram of Your Column Name')
        plt.xlabel('Value')
        plt.ylabel('Density')

        plt.show()

    def load_model (self, focus = 'Location', step = 'Contact', type = 'BreakingBall'):
        model_filename = f'JobLib_Model_Location/joblib_model_{self.focus.name}_{self.currently_modeling}--{self.current_pitch_class}.joblib'
        # self.model.load_model (model_filename)
        self.model = joblib.load (model_filename)
    def generate_predictions (self, focus = 'Location', step = 'Contact', type = 'BreakingBall'):
        # self.focus = Focus.Location
        # self.currently_modeling = step
        # self.current_pitch_class = type
        features = self.features + self.context_features
        # print (features)
        # exit (0)
        # self.current_df = self.input_variables_df
        # self.current_df = self.input_variables_df[
        #     (self.input_variables_df['PitchCall'] == 'InPlay') |
        #     (self.input_variables_df['PitchCall'] == 'StrikeSwinging') |
        #     (self.input_variables_df['PitchCall'] == 'Foul')
        #     ]
        self.current_df['PitchType'] = self.current_df['PitchType'].astype('category')
        self.current_df['PitcherThrows'] = self.current_df['PitcherThrows'].astype('category')
        self.current_df['BatterSide'] = self.current_df['BatterSide'].astype('category')
        # self.input_variables_df['PredictedClass'] = clf.predict(self.input_variables_df[features])
        # self.input_variables_df['PredictedProbability'] = clf.predict_proba(self.input_variables_df[features])[:, 1]
        class_labels = self.model.classes_
        probabilities = self.model.predict_proba(self.current_df[features])
        # Create a new column for each class probability
        for i, class_label in enumerate(class_labels):
            self.current_df[f'Prob_{class_label}'] = probabilities[:, i]
        self.write_current_data(f'{self.focus.name}_{self.currently_modeling}-{self.current_pitch_class}')

# xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb.set_config(verbosity=1)
# xgb_clf.fit(X_train, y_train)
# y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]  # Probability of making contact
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print (roc_auc)
['InPlay' 'FoulBallNotFieldable' 'StrikeCalled' 'BallCalled'
 'StrikeSwinging' 'HitByPitch' 'FoulBall' 'BallinDirt' 'FoulBallFieldable'
 'BallIntentional' 'Undefined' 'WildPitch' 'CatchersInterference'
 'BattersInterference' 'SwinginStrike' 'StirkeCalled']

def train_model (focus=Focus.Location):
    driver = Driver ('radar4.db', 'radar_data', focus)
    # driver.read_radar_data()
    # driver.load_relevant_data()
    # driver.write_variable_data()

    driver.read_variable_data()
    driver.clean_data_for_swing_model()
    driver.clean_data_for_fastballs()
    driver.train_classifier()
    driver.clean_data_for_swing_model()
    driver.clean_data_for_breakingballs()
    driver.train_classifier()
    driver.clean_data_for_swing_model()
    driver.clean_data_for_offspeed()
    driver.train_classifier()

    driver.read_variable_data()
    driver.clean_data_for_take_model()
    driver.clean_data_for_fastballs()
    driver.train_classifier()
    driver.clean_data_for_take_model()
    driver.clean_data_for_breakingballs()
    driver.train_classifier()
    driver.clean_data_for_take_model()
    driver.clean_data_for_offspeed()
    driver.train_classifier()

    driver.read_variable_data()
    driver.clean_data_for_contact_model()
    driver.clean_data_for_fastballs()
    driver.train_classifier()
    driver.clean_data_for_contact_model()
    driver.clean_data_for_breakingballs()
    driver.train_classifier()
    driver.clean_data_for_contact_model()
    driver.clean_data_for_offspeed()
    driver.train_classifier()

    driver.read_variable_data()
    driver.clean_data_for_foul_model()
    driver.clean_data_for_fastballs()
    driver.train_classifier()
    driver.clean_data_for_foul_model()
    driver.clean_data_for_breakingballs()
    driver.train_classifier()
    driver.clean_data_for_foul_model()
    driver.clean_data_for_offspeed()
    driver.train_classifier()

    driver.read_variable_data()
    driver.clean_data_for_in_play_model()
    driver.clean_data_for_fastballs()
    driver.train_classifier()
    driver.clean_data_for_in_play_model()
    driver.clean_data_for_breakingballs()
    driver.train_classifier()
    driver.clean_data_for_in_play_model()
    driver.clean_data_for_offspeed()
    driver.train_classifier()

def run_model (focus=Focus.Location):
    driver = Driver ('radar4.db', 'radar_data', focus)
    # driver.read_radar_data()
    # driver.load_relevant_data()
    # driver.write_variable_data()

    driver.read_variable_data()
    driver.clean_data_for_swing_model(0)
    driver.clean_data_for_fastballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_swing_model(0)
    driver.clean_data_for_breakingballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_swing_model(0)
    driver.clean_data_for_offspeed()
    driver.load_model()
    driver.generate_predictions()

    driver.read_variable_data()
    driver.clean_data_for_take_model(0)
    driver.clean_data_for_fastballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_take_model(0)
    driver.clean_data_for_breakingballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_take_model(0)
    driver.clean_data_for_offspeed()
    driver.load_model()
    driver.generate_predictions()

    driver.read_variable_data()
    driver.clean_data_for_contact_model(0)
    driver.clean_data_for_fastballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_contact_model(0)
    driver.clean_data_for_breakingballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_contact_model(0)
    driver.clean_data_for_offspeed()
    driver.load_model()
    driver.generate_predictions()

    driver.read_variable_data()
    driver.clean_data_for_foul_model(0)
    driver.clean_data_for_fastballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_foul_model(0)
    driver.clean_data_for_breakingballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_foul_model(0)
    driver.clean_data_for_offspeed()
    driver.load_model()
    driver.generate_predictions()

    driver.read_variable_data()
    driver.clean_data_for_in_play_model(0)
    driver.clean_data_for_fastballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_in_play_model(0)
    driver.clean_data_for_breakingballs()
    driver.load_model()
    driver.generate_predictions()
    driver.clean_data_for_in_play_model(0)
    driver.clean_data_for_offspeed()
    driver.load_model()
    driver.generate_predictions()
    return driver

def generate_Location_ratings (driver = Driver ('radar4.db', 'radar_data', Focus.Location)):
    # driver.read_variable_data ()
    # driver.load_predictions ()
    driver.read_predictions (Focus.Location)
    driver.calculate_run_values_swing()
    driver.write_predictions ();
    #
    driver.read_predictions(Focus.Location)
    driver.calculate_average_xRVs()
    driver.read_predictions(Focus.Location)
    driver.write_predictions_players()
    driver.write_players()

    driver.calculate_percentiles()
    driver.write_percentiles()
    # driver.table_to_excel ("Pitcher_Location_Ratings_20_80_scale")

# train_model()
# print (Focus.Location.name)
# exit (0)
# run_model(Focus.Location)
# run_Location_model()
generate_Location_ratings()
driver = Driver ('radar4.db', 'radar_data', Focus.Location)
# driver.read_variable_data()
# driver.classify_pitches()
# driver.write_variable_data()
# driver.read_radar_data()
# driver.read_variable_data()
# driver.write_variable_data()
#TODO: Pitch Classification
#VAA above average
#Bat speed?

# print(tf.__version__)
# run_model ()
# generate_Location_ratings()
# driver.find_overall_percentiles()
# generate_Location_ratings()
# driver.load_model(step = 'InPlay', type = 'Fastball')
# driver.read_variable_data()
# driver.clean_data_for_in_play_model()
# driver.clean_data_for_fastballs()
# driver.generate_predictions()
# exit (0)
# run_model()
# driver.read_radar_data()
# driver.load_relevant_data()
# driver.write_variable_data()
# driver.read_variable_data()
# driver.clean_data_for_contact_model()
# driver.clean_data_for_fastballs()
# driver.train_classifier()
# driver.read_predictions(Focus.Location)
# driver.calculate_average_xRVs()
# driver.write_predictions(Focus.Location)
# driver.calculate_percentiles()
# driver.write_percentiles()
#
# driver.write_players()
# driver.calculate_average_xRVs()
# driver.write_players(Focus.Location)
# driver.table_to_excel("Percentiles_Location_Pitchers")
# driver.calculate_percentiles(Focus.Location)
# driver.write_percentiles(Focus.Location)
# driver.createPlots()
# driver.table_to_excel ("Pitcher_Location_Ratings_20_80_scale")
# driver.read_variable_data()
# driver.write_variable_data()
# driver.load_predictions()
# driver.calculate_run_values_swing()
# driver.write_predictions()
# driver.read_predictions(Focus.Location)
# driver.write_predictions(Focus.Location)
# exit (0)
# driver.calculate_run_values_swing()
# driver.write_predictions ();
# driver.calculate_average_xRVs()
# driver.write_players()
# driver.table_to_excel ("Pitcher_Location_Ratings_20_80_scale")
# driver.read_radar_data()
# driver.load_relevant_data()
# driver.write_variable_data()

def process_data ():
    driver = Driver ('radar4.db', 'radar_data', Focus.Location)
    driver.read_radar_data()
    driver.calculate_inferred_spin_axis()
    driver.calculate_seam_shifted_wake()
    driver.calculate_spin_efficiency()
    driver.normalize_VAA()
    # driver.classify_pitches()
    driver.calculate_average_fastball('Four-Seam')
    driver.calculate_average_fastball('Sinker')
    driver.calculate_average_fastball('Cutter')
    driver.aggregate_fastball_data()
    driver.write_radar_data()
    driver.load_relevant_data()
    driver.write_variable_data()

# driver.read_radar_data()
# driver.load_relevant_data()
# driver.write_variable_data()
# run_model()
# generate_Location_ratings()
