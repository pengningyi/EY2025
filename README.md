# EY2025

This repo is for the 2025 EY Open Science AI & Data Challenge: Cooling Urban Heat Islands. https://challenge.ey.com/2025

We are a team of three, all from Aalto University.
Team member: Ningyi Peng, Wanxian Chen, Afrin Turna

## Goal
1. (Basics) Predict heat island hotspots
2. (Advanced) Find the key factors of hotspots

## Data

### Feature Datasets
- 1. European Sentinel-2 optical satellite data
- 2. NASA Landsat optical satellite data

## Additional Datasets
- 3. Building footprints of the Bronx and Manhattan regions
- 4. Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021

## Validation Datasets
- To predict UHI index on the locations identified in the validation dataset


## Model
- input: location
- output: UHI index

## Tips
- More bands
- Using larger time period than just a day's
- Create a buffer to reduce the error from spatial autocorrelation

## Past year challenge

#### 2024 - IMAGE CLASSIFICATION PROBLEM

https://challenge.ey.com/challenges/tropical-cyclone-damage-assessment-lrrno2xm/overview
This is the link for 2024's challenge.

Here's some solution to 2024's challenge:
https://github.com/Double-Y-EY-Challenge-2024 (Top 10)
https://github.com/kazuma2002/EY-2024-Coastal-Resilience
https://github.com/Marlup/2024-EY-challenge-Coastal-Resilience-Phase-1

Tips from it
- data enhancing
- weather
- image data

#### EY 2022
https://challenge.ey.com/past-challenges
link for 2022's challenge

solution of 1st place winner:
https://github.com/maze508
https://medium.com/@maze508/global-winner-better-working-world-data-challenge-2022-level-3-frog-counting-tool-adb5b495efbe

## Timeline

1st meeting: Feb 2nd

2st meeting: Feb 10th

3rd meeting: Feb 17th 8pm.

4th meeting: Feb 21st, 4pm.


# Implementation
- install the required packages using `pip install -r requirements.txt`.
- For running the code, `python3 main.py --model-type XGBoost`

# Training


## Models
- xgb
- mlp
- wide_deep
- tabnet

## Loss function
Since the methodology used to determine the scores on the leaderboard is the R-Squared score, 

$$\R^2 = 1 - \frac{\sum(y_{true} - y_{pred})^2}{\sum(y_{true} - \bar y_{true})^2}$$

so we use MAE and MSE as Loss func. Huber Loss as well.



# Scoring
- For baseline model, we have 0.5108
- For all 1s, we have 0
- For all random normal ~ (1, 0.1), we have -36


## Schedule

- combine everything together
- 

