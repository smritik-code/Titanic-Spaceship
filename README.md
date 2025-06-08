# Spaceship Titanic: Kaggle Competition Solution

This repository contains a solution for the Kaggle Spaceship Titanic competition. The solution is implemented in Python using pandas and scikit-learn.

## Overview

The Spaceship Titanic competition challenges participants to predict whether passengers were transported to an alternate dimension during the voyage. This repository provides an end-to-end solution, including data preprocessing, feature engineering, model training, and generating predictions.

## Dataset

- **Train Dataset**: `../input/spaceship-titanic/train.csv`
- **Test Dataset**: `../input/spaceship-titanic/test.csv`

## Features Used

- **Cabin**: Encoded categorical feature representing the passenger's cabin.
- **Age**: Numerical feature representing the passenger's age.
- **CryoSleep**: Boolean feature indicating whether the passenger was in cryosleep.
- **Expenses**: Total sum of RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck expenses.

## Preprocessing Steps

1. Removed irrelevant features (`Name`, `PassengerId`, `VIP`, `Destination`).
2. Handled missing values by filling them with `0`.
3. Combined individual expense columns into a single `expenses` feature.
4. Encoded categorical features (`HomePlanet`, `CryoSleep`, `Cabin`).

## Model

- **Random Forest Classifier**:
  - **n_estimators**: 500
  - **max_depth**: 10
  - **random_state**: 50




