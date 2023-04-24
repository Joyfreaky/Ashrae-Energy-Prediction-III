# Ashrae-Energy-Prediction-III
Machine learning pipeline for the Ashrae energy prediction 3 Kaggle competition

## Introduction
Significant investments are being made to improve building efficiencies to reduce costs and emissions. The question is, are the improvements working? That’s where you come in. Under pay-for-performance financing, the building owner makes payments based on the difference between their real energy consumption and what they would have used without any retrofits. The latter values have to come from a model. Current methods of estimation are fragmented and do not scale well. Some assume a specific meter type or don’t work with different building types.

In this competition, you’ll develop accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

About the Host

Founded in 1894, ASHRAE serves to advance the arts and sciences of heating, ventilation, air conditioning refrigeration and their allied fields. ASHRAE members represent building system design and industrial process professionals around the world. With over 54,000 members serving in 132 countries, ASHRAE supports research, standards writing, publishing and continuing education - shaping tomorrow’s built environment today.

## How to Run the Codes. 

Run the terminal command "pip install -r requirements.txt"
# Import file from Kaggle 

1. Follow the Instruction in the Notebook/File_import_api_git.ipynb
 
# Data Preprocessing 


1. Run python command "python src/Memory_Management.py" 
2. Run python command "python src/K-fold_LightGBM.py" this command also implements the Light GBM Model.

# Implementation

1. Implement Random Forest model run "python src/Random_Forest.py" or see Notebook/Random_Forest.ipynb jupyter Notebook
2. Implement CNN model run "src/CNN_Best_Feature.py" or see Notebook/CNN_Best_Feature.ipynb jupyter Notebook

# Data Exploration
1. See Jupyter Notebook after Data Preprocessing step, Notebook/EDA_Leanne.ipynb and Notebook/Krishna_Energy_Analysis NEW.ipynb

# Submission 

1. Last line of code in Notebook/File_import_api_git.ipynb

