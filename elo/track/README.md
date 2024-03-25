“_Well, this year I’m told the team did well because one pitcher had a fine curve ball. I understand that a curve ball is thrown with a deliberate attempt to deceive. Surely this is not an ability we should want to foster at Harvard._” - A quote commonly attributed to Charles Eliot, President of Harvard, but in actuality was likely sourced from Charles Eliot Norton in 1884, Professor of the History of Art at Harvard (Hershberger, 2017).

A good century and then some has passed since this quote entered the annals of history, and times sure have changed. Fastballs only ever get faster, sliders only slide more, and every pitcher now has an arsenal of secondary pitches relying primarily on deception.

This project (accessible [here](https://pitchgrader.streamlit.app/)) represents a white-box (or at least a box which is not totally black) effort to assess collegiate pitchers based off of the intrinsic quality of their pitches, opposition quality notwithstanding. Significant inspiration was taken from Cameron Grove's work on PitchingBot and Professor Alan Nathan's various papers on the physics of baseball. All data was taken from trackman, and the methodology is detailed below.

Note: some files were ommitted from this directory, like the raw .csvs and .db files used for the models. If you for some reason clone this repo, you'll have to add the csvs to your directory, and then run the code in database_driver to create the database files from them.

Pitch Classification: The manually tagged pitch type was used for most pitches. For "corrupted" tagged pitches, the trackman autotagging tool was generally used. Unfortunately, the trackman auto-tagging tool is... not very good, so some efforts were made to correct or otherwise remove rows relying on auto pitch type. Cluster-based classification represents low hanging fruit for future improvement, but an issue is that global clustering is not viable and the most efficient solution I've thought of so far is to cluster on a pitcher by pitcher basis and then fit a classifier on a subset of "representative" pitchers (hierarchical clustering has also seemed to struggle globally).

Stuff Model: The stuff model is split into three submodels - contact, foul, and in play, and further divided into a fastball model, breaking ball model, and offspeed model. No attempt is made to fit the stuff model on takes - stuff in this context purely represents the "nastiness" of a pitch. With this in mind, I did not incorporate the count. Gradient boosted decision trees were used to fit the submodels. Once each submodel was fit, it was used on the entire set of pitches thrown to generate probabilities for each outcome. Those probabilities were used to generate run values, those run values were normalized, and then pitchers were given stuff grades based on their expected run values. The stuff model assumes average command. All percentile sliders on the streamlit deployment are based off of the model's generated probabilities. The full set of features used are: \['PitchType', 'PitcherThrows', 'BatterSide', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'SpinEfficiency*', 'AxisDifference*', 'RelHeight', 'RelSide', 'Extension', 'VAA*'\] for fastballs, \['PitchType', 'PitcherThrows', 'BatterSide', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'SpinEfficiency*', 'AxisDifference*', 'RelHeight', 'RelSide', 'Extension', 'DifferenceRS*', 'DifferenceIVB*', 'DifferenceHB*'\] for breaking balls and offspeeds (* ~> calculated, otherwise taken directly from trackman).

- On some of the more notable features: Axis Difference is an attempt to capture non-Magnus movement and measures the difference between the
  inferred spin axis, which was calculated as        
  self.radar_df['InferredSpinAxis'] = np.where(self.radar_df['pfxx'] < 0,
  (np.arctan(self.radar_df['pfxz'] / self.radar_df['pfxx']) * 180 / math.pi + 90) + 180,
  np.arctan (self.radar_df['pfxz'] / self.radar_df['pfxx']) * 180 / math.pi + 90),
  and the spin axis given by trackman. VAA is, as far as I can tell, not usually used for public stuff models because it is heavily dependent on
  location. However, VAA has a linear relation with plate height, so a normalized version was used, since fastball shape is such a big piece of its effectiveness.

Location Model: Similar to the stuff model, except with 2 extra submodels, take and swing. Assumes average stuff for each pitcher. The full set of features used are: 'PitchType', 'PitcherThrows', 'BatterSide', 'Balls', 'Strikes', 'PlateLocHeight', 'PlateLocSide'.

Swing Mechanics: the batters were in the database, so I had to do something with them, right? Some notable outputs of the model are:

- Collision Coefficient - taken from Professor Alan Nathan's research, collision coefficient represents how efficiently the bat was able to
  convert the speed of the ball coming in and the speed of the bat into exit velocity. It scales with distance from the sweet spot, meaning the    collision is most efficient in the sweet spot of the bat (I know, hard to believe) and decreases as you travel towards the tips. Collision     
  coefficient has a linear relationship with exit velocity, so I ran a regression to estimate the collision coefficient. This assumes that the
  bat speed is constant, so to mitigate the effects I regressed individually for each batter, assuming that their top evit velos were produced
  by striking the ball in the sweet spot. It still ended up a bit overfit with exit velo, but that probably won't be a problem, right?

- (Effective) Bat Speed - also taken from Professor Alan Nathan's research. It turns out, CC being overfit with EV is a problem after all. Once you have the CC of each batted ball event you can calculate bat speed, but it ends up underestimating speeds for low EV players.

- True Bat Speed - by taking only "barrels" (https://www.mlb.com/glossary/statcast/barrel) into account, we can assume that the ball impacted
  the bat somewhere close to the sweet spot, sidestepping the issue of calculating CC completely. Really, "True Bat Speed" would also need the     attack angle to match the launch angle, but I don't think I'd have enough data to make a stable measurement for more than a handful of       
  batters. Probably low hanging fruit for the future.

- Attack Angle - the average vertical angle of the bat when it impacts the ball. EVs are maximized when attack angle equals centerline angle,    
  and furthermore AA = LA when LA = CA. So, AA was calculated by fitting a parabola to the highest EVs bucketed into LAs of a player's BBEs.

- "Smash Factor" - a confusingly named stat form Driveline which is just balls in play divided by fouls and whiffs weighted by collision
  coefficient. It's stickier than zone contact and K%, so it is their "bat to ball" skill marker.

- Contact Quality - intrinsic run value of ball when contact is made

- Swing Decision - calculated by fitting a model to predict swings based on the difference between the expected intrinsic value of a swing vs. the run value of a take. Final metric is based off how often they choose "correctly" compared to the league. It's a little funky right now, and I need to tinker with the model a little.

**References**:

* Hershberger, R. (2017). With a Deliberate Attempt to Deceive: Correcting a Quotation Misattributed to Charles Eliot, President of
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Harvard. Baseball Research Journal, Spring 2017.
