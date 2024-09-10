# Titanic - Machine Learning from Disaster

This project is part of the Kaggle competition "Titanic: Machine Learning from Disaster". The goal is to predict the survival of passengers on the Titanic using machine learning techniques.

## Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

In this challenge, we aim to build a predictive model that answers the question: "what sorts of people were more likely to survive?" using passenger data (i.e., name, age, gender, socio-economic class, etc.).

## Dataset

The dataset used in this project is provided by Kaggle and consists of two files:
- `train.csv`: The training set containing the features and the target variable (Survived).
- `test.csv`: The test set containing the features without the target variable.

## Features

The dataset contains the following features:
- `PassengerId`: Unique ID for each passenger.
- `Survived`: Survival (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Name of the passenger.
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard the Titanic.
- `Parch`: Number of parents/children aboard the Titanic.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number.
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Model and Evaluation

The model used for this project is a Random Forest Classifier. The model was trained on the training dataset and evaluated using cross-validation. The final model achieved a score of 0.76 on the Kaggle leaderboard.

## Installation

To run this project, you need to have Python installed along with the following libraries:
- random
- missingno
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- fasteda

You can install the required libraries using pip:
```sh
pip install missingno numpy pandas matplotlib seaborn scikit-learn fasteda
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/EngJurado/titanic-machine-learning-from-disaster.git
```

2. Navigate to the project directory:

```sh
cd titanic-ml-from-disaster
```

3. Run the Jupyter Notebook:

```sh
jupyter notebook Titanic_Machine_Learning_from_Disaster.ipynb
```

## Results

The final model achieved a score of 0.76 on the Kaggle leaderboard. The model's performance can be further improved by feature engineering, hyperparameter tuning, and using more advanced machine learning algorithms.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Kaggle for providing the dataset and the platform for the competition.
- The Kaggle community for their valuable discussions and insights.
