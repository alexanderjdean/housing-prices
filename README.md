# housing-prices

This is a project that uses the [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle dataset to predict housing prices and generate a submission for Kaggle's competition.

The Kaggle API is used to download the dataset and preprocess the data by removing null values and one hot encoding categorical variables. Then the following regression models are built: Random Forest, Support Vector Machines, Decision Trees, and XGBoost with accompanying accuracy rates based on the test data.

The ```generate_submission``` function in ```util.py``` formats the output to comply with Kaggle's competition entry requirements.