# COVID-19 Case Severity Analysis   
   
## Introduction    
We are going to be analyzing how different factors affect the severity of COVID-19 in an infected patient.
We will be looking primarily at pre-existing diseases, as this is theorized to increase the chance of
serious illness or even death from coronavirus. In addition, we will also take into account how
demographic factors like age, race, sex, and financial status affect the severity.
We know that having a pre-existing disease will increase the chance of medical problems related to coronavirus,
but we do not know how much each disease affects the severity. We hope this intersection of
pre-existing diseases and demographics with severity of illness in COVID-19 patients will lead to insightful information about the virus.    

## Methods

### Unsupervised Learning
In any machine learning task, the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) is
something that must be dealt with.
What this means is that as the dimensionality (or number of features in the dataset) increases,
the volume of the feature space in which we're working increases rapidly.
As a result of this, it becomes necessary to reduce the dimension into a context that we can more easily work with.
By reducing the amount of dimensions/features we have to process, we are able to analyze fewer relationships between
features and reduce the computational power needed and the likelihood of overfitting.

#### [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
An important algorithm in dimensionality reduction is principal component analysis (PCA),
which is an unsupervised technique.
Principal component analysis works by projecting higher-dimensional spaces to lower-dimensional bases, where the new
bases correspond to the most important components of the data (hence the name principal component analysis).
Essentially, the goal of PCA is to reduce the amount of dimensions as much as possible
while keeping as much information as possible.

### Supervised Learning:
The point of any machine learning task is to get some actionable results out of the data that we put in,
and supervised learning will help us achieve that goal.
The main methods being considered are decision trees and regression.

#### [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree)
<p align="center">
  <img src="https://miro.medium.com/max/1000/0*YlmscpNST8MY7yzw.png" width = 50%/>
</p>
A decision tree is a tree-like structure which models the effects of
certain actions that lead to certain possibilities.
The goal of a decision tree is to split the classification of a certain input into groups
through a process of sequential questions or conditions.
In our case, these conditions might regard which underlying conditions individuals have.
These conditions could sort examples into different groups.

#### [Regression](https://en.wikipedia.org/wiki/Regression_analysis)
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png" width = 50%/>
</p>
Regression analysis is a way to approximate the relationship between our dependent variable
and our independent variables.
The most basic form of regression is univariate linear regression, in which one tries to fit a straight line
that captures the relationship between one dependent and one independent variable.
In our case, we'll be trying to build an algorithm that learns the relationships between some continuous, numerical
output and the input features (different attributes of patients).
By using regression, we could be able to return some percentage chance of developing a serious illness
based on the combination of COVID-19 with other risk factors (our inputs).

## **Results**
Our ideal results will show a clear relation between severity and various demographic backgrounds.
The principal component analysis will ideally identify the comorbidities that lead to the highest severity. 
For example, we may find that COVID-19 in conjunction with heart disease may be more likely to result
in a severe case in comparison to COVID-19 in conjunction with diabetes. For the supervised portion of the project,
we expect our model to accurately predict discrete categories of severity in our training set based on other relevant patient information.

## Metrics
For this project we used a variety of metrics to find the ideal hyperparameters for our model. We primarily measured 4 different metrics for performance: the F1 score, precision, recall, and accuracy. It is important to note that for our classification problem, there were only 2 classes to choose from. Every patient was either in Death_ICU or not in Death_ICU. This means that randomly picking between the two categories could yield an accuracy around 50%. In addition, our dataset had 116 total rows with 48 of them being in the Death_ICU category and 68 not being in Death_ICU. This also means that a classifier that picked every data point as false would be able to get around 58.6% accuracy.

# Precision: 
Precision is a measure of how correctly the returned positives were predicted. In order to calculate the precision, we divide the number of correctly predicted positives by the total number of predicted positives. Maximizing the precision will decrease the number of false positives (actual negatives that were classified as positive) the classifier returns. 

# Recall: 
Recall is a measure of how much of the actual positives were returned as positive. To calculate the recall we divide the number of correctly predicted positives by the total number of positives in the dataset. Maximizing the recall will decrease the number of false negatives (actual positives that were classified as negative) the classifier returns.

# Accuracy:
Accuracy is a measure of the correctness of the classifier. In order to calculate the accuracy, we add the number of correctly predicted positives with the number of correctly predicted negatives and divide that by the total number of rows in the dataset.


# F1-Score:
The F1 score is another way to measure the accuracy of a classifier. F1 combines both precision and recall into one metric. We calculate the F1 score by multiplying our precision and recall together and then multiplying by 2 and then dividing by the sum of our precision and recall. Maximizing the F1 score will reduce both the number of false positives and the number of false negatives the classifier returns. However one downside of the F-score is that it does not take into account the number of True Negatives (actual negatives that were correctly returned as negative). 

In a real-world setting, where we are attempting to measure the potential severity of a patient with COVID-19, the number of true negatives is not as important as the number of false positives or the number of false negatives. For example, telling patients who will never need to go to the ICU that they will soon be in the ICU (False Positive) and keeping them in the hospital for continuous monitoring will take away much needed hospital beds from patients who desperately need them. On the other hand, telling patients that will fall severely ill and will soon need to go to the ICU that they are fine (False Negative) and sending them home early will also divert hospital resources from those who need it most. For this reason, we decided to focus primarily on maximizing the F1-score which accounts for both False Positives and False Negatives.


## **Discussion**
Predicting risk based on demographic information, medical background, and behavior can provide extremely valuable insight
into how the COVID-19 pandemic should best be handled. At the institutional level, hospitals can use our risk predictions
to determine how to most efficiently allocate the limited resources in order to minimize deaths and complications.
Hospitals will be able to make well-informed, data-driven decisions for how to treat patients and what to be the most wary of.
Moreover, risk prediction and a strong understanding of what factors contribute the most to COVID-19 severity can also be informative for the individual.
An individual may engage in more extensive prevention behaviors if they are able to predict the severity of their illness or the illnesses of their loved ones.
Additionally, as a society, we can identify those individuals who are most at risk, and take extra precautions to protect them from the virus.
We hope that this increase in information will drive progress toward ending the pandemic.
