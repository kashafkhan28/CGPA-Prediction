## CGPA Prediction Project

### Overview
This project aims to develop a machine learning system to predict the final Cumulative Grade Point Average (CGPA) of a student at the end of their fourth year, based on the Grade Points (GPs) obtained in the initial years (up to first, second, or third year). The dataset used for training and evaluation is provided as "The_Grades_Dataset.csv".

### Dataset
The dataset contains the following columns:
- `First_Year_GP`: Grade Points obtained in the first year courses.
- `Second_Year_GP`: Grade Points obtained in the second year courses.
- `Third_Year_GP`: Grade Points obtained in the third year courses.
- `Fourth_Year_GP`: Grade Points obtained in the fourth year courses.
- `Final_CGPA`: The target variable, the final CGPA of the student at the end of the fourth year.

### Model Descriptions
#### Model 1: Predict final CGPA based on GPs of the first year only using linear regression.
#### Model 2: Predict final CGPA based on GPs of the first two years using linear regression.
#### Model 3: Predict final CGPA based on GPs of the first year only using random forest regressor.
#### Model 4: Predict final CGPA based on GPs of the first two years using random forest regressor.

### Installation
1. Clone this repository to your local machine.
2. Ensure you have Python 3.x installed on your system along with the necessary libraries: NumPy, Pandas, and Scikit-learn.
3. Navigate to the project directory.
4. Place the dataset file "The_Grades_Dataset.csv" in the project directory.

### Contributions
Contributions are welcome! Feel free to submit issues or pull requests if you have any suggestions for improvements or encounter any bugs.
