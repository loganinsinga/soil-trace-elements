# Soil Trace Element Cycling

This repository contains scripts that support the publication titled "Widespread losses of essential elements predicted for European soils". 

## Table of Contents

- Introduction
- Description
- Installation
- Usage
- License

## Introduction

These scripts were used to perform parts of the machine learning, statistical analysis, and figure creation for this work.
This repository is submitted for transparency and so that others may use it as a resource.

## Description

The scripts in this repository orgnized into 4 steps of the workflow.
1. Transforms the predictor variables to make distributions more normal.
2. Z-scores (standardizes) the predictor variables.
3. Tunes, evaluates, and executes the machine learning models to generate predictions. One script for each model-element pair (e.g., MLP-S)
4. Generates maps and charts of the model predictions and statistics.

## Installation

To use the scripts in this repo, follow these steps:

1. Clone this repository to your local development system
2. Install the dependences
3. Set up your local Python environment

## Usage

Each script is executed individually in a step-wise sequential process.
Any user-defined inputs are defined as constants that can be altered.
The input to the first script is the master table (Master_table.xlsx).

## Licence

Copyright © 2024 EcoChem Lab.

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

A copy of the license is available in the repository's [LICENSE](LICENSE) file.
