import numpy as np

import database_driver
import stuff_plus
import dash
import streamlit as st
import pandas as pd

# st.write("""
# # My first app
# Hello *world!*
# """)


def interpolate_color(minval, maxval, val, color_palette):
    """Interpolate between colors in color_palette where color_palette
       is an array of tuples/lists with 3 integers indicating RGB values."""
    max_index = len(color_palette)-1
    v = float(val-minval) / float(maxval-minval) * max_index
    i1, i2 = int(v), min(int(v)+1, max_index)
    r1, g1, b1 = color_palette[i1]
    r2, g2, b2 = color_palette[i2]
    f = v - i1
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def color_for_value(value):
    # Define the color transition [0%, 50%, 100%]
    colors = [(26, 28, 244), (128, 128, 128), (255, 25, 25)]  # Blue, Grey, Red
    r, g, b = interpolate_color(0, 100, value, colors)
    return f'#{r:02x}{g:02x}{b:02x}'
def display_static_slider(label, value, max_value=100.0):
    # Calculate the percentage position of the value
    if (value != value):
        value = -1
    percentage = value
    value_str = str(int(value))
    min_width = 35  # This is the minimum width for 1-2 digit numbers
    extra_width_per_digit = 10  # Additional width for each extra digit
    bubble_width = min_width + (len(value_str) - 2) * extra_width_per_digit
    if len(value_str) < 2:
        bubble_width = min_width
    color = color_for_value(value)
    value = int (value)
    # percentage = "{:.0f}%".format(percentage)

    # Create the "slider" using markdown with custom styling
    # red #ff4b4b, yellow (255, 221, 125), darker red (255, 25, 26)
    slider_bar = f"""
        <div style='width: 100%; padding-bottom: 1rem;'>  <!-- Increase padding-bottom as needed -->
        <div style='text-align: center;'>{label}</div>
        <div style='margin-top: 1rem; background-color: #f2f2f2; border-radius: 10px; height: 10px; position: relative;'> <!-- Adjust margin-top to increase space -->
            <div style='position: absolute; top: -20px; left: {percentage}%; transform: translateX(-50%);'>
                <div style='background-color: {color}; color: white; padding: 0.5rem; border-radius: 20px; width: {bubble_width}px; text-align: center;'>
                    {value}
                </div>
            </div>
        </div>
    </div>
    """


    # Display the slider bar
    st.markdown(slider_bar, unsafe_allow_html=True)

# Example of displaying the static sliders
# col1, col2 = st.columns(2)
#
# # Display static sliders within these columns
# # display_static_slider('Overall xRun Value', 97)
# # display_static_slider('xSwStr%', 97)
# # display_static_slider('xK%', 91)
# # display_static_slider('xGB%', 92)
# with col1:
#     st.markdown(display_static_slider('Overall xRun Value', 97), unsafe_allow_html=True)
#     st.markdown(display_static_slider('xSwStr%', 97), unsafe_allow_html=True)
#
# with col2:
#     st.markdown(display_static_slider('xK%', 91), unsafe_allow_html=True)
#     st.markdown(display_static_slider('xGB%', 92), unsafe_allow_html=True)

st.title('Stuff+ Model (also a swing mechanics model now)')
st.write('Database last updated 3/21/2024')
# Create two text input boxes for the first and last name
if 'team_flag' not in st.session_state:
    st.session_state.team_flag = False

# Your database initialization
driver = database_driver.DatabaseDriver()
stuff_driver = stuff_plus.Driver('radar2.db', 'radar_data')
# Update dataset button
col1, col2, space = st.columns([2, 2, 2])
with col1:
    update = st.button("Update Dataset", key='update_dataset', type = 'primary')
    if update:
        st.write (f"I'm just a placeholder button")
        # st.write('Updating. May take a while')
        # driver.read_data()
        # stuff_plus.process_data()
        # stuff_plus.run_model()
        # stuff_plus.generate_stuff_ratings()
        # driver.write_data()
        # st.write('Updated. You may have to reload the page to see the effects')
# Button to toggle between personal details and team view
with col2:
    team_toggle = st.button("Toggle team/player")
    if (team_toggle):
        st.session_state.team_flag = not st.session_state.team_flag
        # st.write (team_flag)

batting_percentiles_df = driver.retrieve_percentiles_bat_team ('All')
pitching_stuff_df = driver.retrieve_stuff_team ('All')
# batting_names_raw = batting_percentiles_df ['Batter']
# pitching_names_raw = pitching_stuff_df ['Pitcher']
# combined_names_raw = pd.concat([batting_names_raw, pitching_names_raw])
# Extract names and teams, and add a label for type

batting_percentiles_df ['Role'] = 'Batter'
batting_percentiles_df ['Name'] = batting_percentiles_df ['Batter']
batting_percentiles_df ['Team'] = batting_percentiles_df ['BatterTeam']
pitching_stuff_df ['Role'] = 'Pitcher'
pitching_stuff_df ['Name'] = pitching_stuff_df ['Pitcher']
pitching_stuff_df ['Team'] = pitching_stuff_df ['PitcherTeam']
combined_df = pd.concat([batting_percentiles_df, pitching_stuff_df])
combined_df['Name_Sorted'] = combined_df['Name'].str.lstrip()
combined_df = combined_df.sort_values(by='Name_Sorted')
combined_df.reset_index(drop=True, inplace=True)
combined_names = combined_df.apply(lambda x: f"{' '.join(x['Name'].split(', ')[::-1])} - {x['Team']}, {x['Role']}", axis=1)
combined_dict = dict(zip(combined_names, combined_df['Name']))

# batting_names = batting_percentiles_df.apply(lambda x: f"{' '.join(x['Batter'].split(', ')[::-1])} - {x['BatterTeam']}, Batter", axis=1)
# # batting_dict = {f"{' '.join(x['Batter'].split(', ')[::-1])} - {x[f'BatterTeam']}, Batter": x['Batter'] for _, x in batting_percentiles_df.iterrows()}
# pitching_names = pitching_stuff_df.apply(lambda x: f"{' '.join(x['Pitcher'].split(', ')[::-1])} - {x['PitcherTeam']}, Pitcher", axis=1)
# # pitching_dict = {f"{' '.join(x['Pitcher'].split(', ')[::-1])} - {x[f'PitcherTeam']}, Pitcher": x['Pitcher'] for _, x in pitching_stuff_df.iterrows()}
# batting_dict = dict(zip(batting_names, batting_percentiles_df['Batter']))
# pitching_dict = dict(zip(pitching_names, pitching_stuff_df['Pitcher']))
# combined_dict = {**batting_dict, **pitching_dict}

pitching_stuff_df = pitching_stuff_df [pitching_stuff_df ['PitcherTeam'] != 'VIR_CAV']
team_names = sorted (pitching_stuff_df ['PitcherTeam'].unique().tolist())
options_teams = ['', 'All', 'VIR_CAV'] + team_names
# Combine into a single series
# combined_names = pd.concat([batting_names, pitching_names]).sort_values().unique()

# options = list(zip(combined_names, combined_names_raw))
# options_sorted = sorted(options, key=lambda x: x[1])
options = [''] + list(combined_names)

# batting_percentages_df = driver.retrieve_percentages_bat_team ('All')
# pitching_percentages_df = driver.retrieve_percentages_team ('All')
# pitching_stuff_df = driver.retrieve_stuff_team ('All')

# Conditional rendering based on the toggle state
if not st.session_state.team_flag:
    # first_name = st.text_input('First Name', '', placeholder='First name', key='first_name')
    # last_name = st.text_input('Last Name', '', placeholder='Last name', key='last_name')
    # team_name = st.text_input('Team Name', '', placeholder='Team name', key='team_name')
    selected_name = st.selectbox('Player', options=options, key='player_name')
    team_name = ''
    # When both names have been entered, display the full name
    display_name = st.empty()
    # if first_name and last_name:
    if selected_name != '':
        # display_name = st.empty()
        # display_name.success(f'Player name: {first_name} {last_name}') #want to update this
        # name = last_name + ", " + first_name
        # name_parts = selected_name.split(' - ')[0]
        # name = name_parts.split(' ')[1] + ", " + name_parts.split(' ')[0]
        name = combined_dict.get(selected_name, '')
        # st.success (team_name)
        df = driver.retrieve_percentiles (name, team_name)
        # df = pitching_percentiles_df [pitching_percentages_df ['Pitcher'] == name]
        if (df.empty) or (selected_name.split(', ')[1] == 'Batter'):
            df = driver.retrieve_percentiles_batter(name, team_name)
            # df = batting_percentiles_df [batting_percentiles_df ['Batter'] == name]
            if (df.empty):
                #want to write update here
                # st.error(f"{last_name}, {first_name} not found. Remember that the name is case sensitive. If you're looking for a batter, keep in mind that batters need to have >100 BBE to qualify")
                st.error(f"{name} not found. Something has gone wrong in the backend! Send an email to wsg9mf@virginia.edu")
            else:
                # df = df.drop (columns = ['Batter', 'BatterTeam'])
                raw_df = driver.retrieve_percentages_batter(name)
                # raw_df = batting_percentages_df [batting_percentages_df ['Batter'] == name]
                # raw_df = df
                batter_side_counts = raw_df.groupby(['Batter', 'BatterSide']).size().unstack(fill_value=0)
                batter_side_counts['Total'] = batter_side_counts.sum(axis=1)
                try:
                    batter_side_counts['LeftProp'] = batter_side_counts['Left'] / batter_side_counts['Total']
                    batter_side_counts['RightProp'] = batter_side_counts['Right'] / batter_side_counts['Total']
                    switch_batters = batter_side_counts[(batter_side_counts['LeftProp'] > 0.04) & (batter_side_counts['RightProp'] > 0.04)].index
                    raw_df.loc[raw_df['Batter'].isin(switch_batters), 'BatterSide'] = 'Switch'
                except:
                    print ('not switch hitter')
                # display_name.success (f"Batter: {first_name} {last_name}, {raw_df ['BatterTeam'].iloc [0]}. Bats: {raw_df ['BatterSide'].iloc [0]}")
                display_name.success (f"Batter: {name}. {raw_df ['BatterTeam'].iloc [0]}. Bats: {raw_df ['BatterSide'].iloc [0]}")
                raw_df = raw_df.head (1)
                raw_df = raw_df.drop (columns = ['Batter', 'BatterTeam', 'BatterSide'])
                speed_df = raw_df [['AttackAngle', 'TrueBatSpeed', 'AverageBatSpeed', 'AverageHandSpeed', 'AverageBarrelSpeed']]
                speed_df.rename(columns={'AverageBatSpeed': 'EffectiveBatSpeed', 'AverageHandSpeed': '"HandSpeed"', 'AverageBarrelSpeed' : '"BarrelSpeed"'}, inplace=True)
                raw_df = raw_df.drop (columns = ['AttackAngle', 'TrueBatSpeed', 'AverageBatSpeed', 'AverageHandSpeed', 'AverageBarrelSpeed'])
                container = st.container()
                container.markdown("<div margin-left: auto, margin-right: auto>", unsafe_allow_html=True)
                container.dataframe(speed_df)
                container.dataframe(raw_df)
                container.markdown("</div>", unsafe_allow_html=True)
                index = 0
                def add_custom_css():
                    st.markdown("""
                        <style>
                            .block-container > .row {
                                gap: 2rem;  /* Adjust the gap size as needed */
                            }
                        </style>
                    """, unsafe_allow_html=True)

                add_custom_css()
                col1, space, col2 = st.columns([2, 1, 2])

                with col1:
                    display_static_slider('Bat Speed', df ['TrueBatSpeed'].iloc [index])
                    display_static_slider('Pitch Selection', df ['SwingDecision'].iloc [index])
                    display_static_slider('Contact Efficiency', df ['AverageEA'].iloc [index])
                    display_static_slider('Raw Power', df ['NeutralExitSpeed'].iloc [index])

                with col2:
                    # display_static_slider('Bat-to-Ball', df ['AverageSF'].iloc [index])
                    display_static_slider('Effective Bat Speed', df ['AverageBatSpeed'].iloc [index])
                    display_static_slider('Bat-to-Ball', df ['AverageSF'].iloc [index])
                    display_static_slider('Contact Quality', df ['AverageI'].iloc [index])
                    display_static_slider('"Game Power"', df ['NeutralHR'].iloc [index])

        else:
            stuff_df = driver.retrieve_stuff (name)
            stuff_df = stuff_df.round(0)

            rename_columns = {
                'ChangeUp': 'CH',
                'Curveball': 'CU',
                'Cutter' : 'FC',
                'Four-Seam' : 'FF',
                'Sinker' : 'SI',
                'Slider' : 'SL',
                'Splitter' : 'FS'
            }
            desired_order = ['PitchCount', 'Overall Stuff', 'FF', 'SI', 'FC', 'SL', 'CU', 'FS', 'CH']
            stuff_df = stuff_df.rename(columns=rename_columns)

            # stuff_df = pitching_stuff_df [pitching_stuff_df ['Pitcher'] == name]
            stuff_df = stuff_df.drop_duplicates ('Pitcher')
            stuff_df = stuff_df.drop (columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows'])
            # stuff_df = stuff_df.drop_duplicates ('Pitcher')
            stuff_df.rename(columns={'Overall': 'Overall Stuff'}, inplace=True)
            columns_to_drop = [column for column in stuff_df.columns if column.endswith('Usage')]
            stuff_df = stuff_df.drop(columns=columns_to_drop)
            stuff_df = stuff_df.dropna(axis=1)
            # st.success (desired_order)
            # st.success (stuff_df.columns)
            # actual_order = [col for col in desired_order if col in stuff_df.columns]
            # st.success (actual_order)
            # stuff_df = stuff_df[actual_order]
            # st.markdown("""
            #     <style>
            #     .centered-df {
            #         margin-left: auto;
            #         margin-right: auto;
            #     }
            #     </style>
            #     """, unsafe_allow_html=True)
            # container = st.empty ()
            container = st.container()
            container.markdown("<div margin-left: auto, margin-right: auto>", unsafe_allow_html=True)
            container.dataframe(stuff_df)
            container.markdown("</div>", unsafe_allow_html=True)

            # display_name.success (f"Pitcher: {first_name} {last_name}, {df ['PitcherTeam'].iloc [0]}. Throws: {df ['PitcherThrows'].iloc [0]}")
            display_name.success (f"Pitcher: {name}. {df ['PitcherTeam'].iloc [0]}. Throws: {df ['PitcherThrows'].iloc [0]}")
            df = df.drop_duplicates ('PitchType')
            df = df.drop (columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'Balls', 'Strikes'])
            cols = [col for col in df.columns if col != 'xRV']
            cols.insert(2, 'xRV')
            df = df[cols]
            df = df.sort_values(by='Usage', ascending = False)
            st.dataframe(df)
            pitch_types = df['PitchType'].unique().tolist()
            index = st.selectbox("Pitch Type", range(len(pitch_types)), format_func=lambda x: pitch_types[x])
            temp = df['PitchType'].iloc [index]
            # st.title (temp)
            def add_custom_css():
                st.markdown("""
                        <style>
                            .block-container > .row {
                                gap: 2rem;  /* Adjust the gap size as needed */
                            }
                        </style>
                    """, unsafe_allow_html=True)

            add_custom_css()
            col1, space, col2 = st.columns([2, 1, 2])

            with col1:
                display_static_slider('xRV', df ['xRV'].iloc [index])
                display_static_slider('xWhiff%', df ['xWhiff%'].iloc [index])
                display_static_slider('xFoul%', df ['xFoul%'].iloc [index])

            with col2:
                display_static_slider('xGB%', df ['xGB%'].iloc [index])
                display_static_slider('xHH%', 100 - df ['xHH%'].iloc [index])
                display_static_slider('xHHFB%', 100 - df ['Prob_HardFB'].iloc [index])

            st.write ("View/Edit Raws")
            prob_df = driver.retrieve_percentages(name)
            prob_df = prob_df.drop_duplicates ('PitchType')
            # prob_df = pitching_percentages_df [pitching_percentages_df ['Pitcher'] == name]
            prob_df = prob_df.drop (columns = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'Balls', 'Strikes'])
            cols = [col for col in prob_df.columns if col != 'xRV']
            cols.insert(2, 'xRV')
            prob_df = prob_df[cols]
            prob_df = prob_df.sort_values(by='Usage', ascending = False)
            input_df = st.data_editor(prob_df)
            # update = st.button("Update Percentiles", key='update_percentiles', type = 'secondary')
            #TODO: this
    # df = pd.read_csv("my_data.csv")
    # st.line_chart(df)
else:
    # Here you can add your logic or widgets to display team view
    # team_name = st.text_input('Team ID (from trackman)', '', placeholder='Team ID (UVA is VIR_CAV) - Enter "All" to see all players', key='team_name')
    team_name = st.selectbox('Team ID (UVA is VIR_CAV)', options=options_teams, key='team_name')
    min_pitch = st.text_input('Minimum Pitch Count', '', placeholder='Pitch Count', key='min_pitch')
    display_name = st.empty()
    if team_name != '':
        df = driver.retrieve_percentiles_team (team_name)
        # df = pitching_percentiles_df [pitching_percentiles_df ['PitcherTeam'] == team_name]
        df_bat = driver.retrieve_percentiles_team_bat (team_name)
        # df_bat = batting_percentiles_df [batting_percentiles_df ['BatterTeam'] == team_name]
        if (df.empty):
            #want to write update here
            st.error(f'{team_name} not found. Remember that the name is case sensitive')
        else:
            stuff_df = driver.retrieve_stuff_team (team_name)
            stuff_df = stuff_df.round(0)
            rename_columns = {
                'ChangeUp': 'CH',
                'Curveball': 'CU',
                'Cutter' : 'FC',
                'Four-Seam' : 'FF',
                'Sinker' : 'SI',
                'Slider' : 'SL',
                'Splitter' : 'FS'
            }
            desired_order = ['Pitcher', 'PitcherTeam', 'PitcherThrows', 'PitchCount', 'Overall', 'FF', 'SI', 'FC', 'SL', 'CU', 'FS', 'CH']
            stuff_df = stuff_df.rename(columns=rename_columns)
            # stuff_df = pitching_stuff_df [pitching_stuff_df ['PitchingTeam'] == team_name]
            if team_name != 'All':
                stuff_df = stuff_df.drop (columns = ['PitcherTeam'])
            columns_to_drop = [column for column in stuff_df.columns if column.endswith('Usage')]
            stuff_df = stuff_df.drop(columns=columns_to_drop)
            if min_pitch:  # Check if something was entered
                try:
                    min_pitch = int(min_pitch)
                    stuff_df = stuff_df [stuff_df ['PitchCount'] >= min_pitch]
                except ValueError:
                    st.error("Invalid number for the minimum pitch count.")
            actual_order = [col for col in desired_order if col in stuff_df.columns]
            # st.success (actual_order)
            stuff_df = stuff_df[actual_order]
            container = st.container()
            container.markdown("<div margin-left: auto, margin-right: auto>", unsafe_allow_html=True)
            container.dataframe(stuff_df)
            container.markdown("</div>", unsafe_allow_html=True)

            display_name.success (f"Team: {team_name}")
            if (team_name == 'All'):
                df = df.drop (columns = ['Balls', 'Strikes'])
            else:
                df = df.drop (columns = ['PitcherTeam', 'Balls', 'Strikes'])
                df_bat = df_bat.drop (columns = ['BatterTeam'])
            if min_pitch:  # Check if something was entered
                try:
                    valid_pitchers = stuff_df['Pitcher']
                    df = df[df['Pitcher'].isin(valid_pitchers)]
                except ValueError:
                    print ('hey')
            cols = [col for col in df.columns if col != 'xRV']
            cols.insert(3, 'xRV')
            df = df[cols]
            # df = df.sort_values(by='Usage', ascending = False)
            st.dataframe(df)
            st.dataframe (df_bat)
            # pitch_types = df['PitchType'].unique().tolist()
            # index = st.selectbox("Pitch Type", range(len(pitch_types)), format_func=lambda x: pitch_types[x])
            # temp = df['PitchType'].iloc [index]
            # stuff_df = stuff_df.dropna(axis=1)


