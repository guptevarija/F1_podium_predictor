# 🏎️ F1 Podium Predictor

A machine learning project that predicts Formula 1 podium finishes using historical race performance, driver form, constructor performance, circuit history, and championship context.

The project is designed to answer a practical question:

> **Given the information available before a Formula 1 race, which drivers are most likely to finish on the podium?**

---

## 🎯 Project Objective

The objective is to build a leakage-aware machine learning pipeline for predicting whether an F1 driver will finish in the top three.

The project uses historical Formula 1 data from **2018–2025** to train and evaluate machine learning models and then applies the trained model to predict the **2026 Singapore Grand Prix**.

---

## 📊 Data

The project uses Formula 1 race and qualifying data collected through the Jolpica F1 API.

Historical race data:

- Seasons: 2018–2025
- Race results
- Driver information
- Constructor/team information
- Starting grid
- Finishing position
- Championship points
- Race status
- Circuit information

Qualifying data is used for the post-qualifying version of the model.

---

## 🧠 Features

The model uses several categories of features.

### Driver recent form

- Average finishing position over the previous 5 races
- Average points over the previous 5 races
- Number of podiums in the previous 5 races
- DNF rate over the previous 5 races

### Constructor recent form

- Average team finishing position
- Average team points
- Number of team podiums
- Team DNF rate

### Circuit-specific performance

- Driver's historical Singapore finishing position
- Driver's historical Singapore podiums
- Constructor's historical Singapore performance
- Street-circuit performance

### Championship context

- Driver championship points before the race
- Driver championship position
- Constructor championship points before the race
- Constructor championship position

### Qualifying information

For the post-qualifying model:

- Qualifying position
- Starting grid position

---

## 🔒 Leakage Prevention

A key objective of this project is to avoid using information that would not have been available at prediction time.

For rolling features, the current race is excluded using a one-race shift before calculating rolling statistics.

For example:

```python
x.shift(1).rolling(5)