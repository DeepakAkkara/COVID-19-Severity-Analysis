# **COVID-19 Case Severity Analysis**
<p align="center">
    <img src="assets/Design%20Process.png" width=50%/>
</p>

## **Introduction**
We are going to be analyzing how different factors affect the severity of COVID-19 in an infected patient.
We will be looking primarily at pre-existing diseases, as this is theorized to increase the chance of
serious illness or even death from coronavirus. In addition, we will also take into account how
demographic factors like age, race, sex, and financial status affect the severity.
We know that having a pre-existing disease will increase the chance of medical problems related to coronavirus,
but we do not know how much each disease affects the severity. We hope this intersection of
pre-existing diseases and demographics with severity of illness in COVID-19 patients will lead to insightful information about the virus.    


## **Background**
Hospitals have been overcrowded with COVID patients since the pandemic started.
COVID is a deadly virus that has killed over 1 million worldwide and 208,000 people in the United States,
and these numbers will continue to increase.
Too many people are dying, so we need to minimize these deaths as much as possible by prioritizing beds for the most vulnerable.  
Our goal is to produce an algorithm that can assign patients a severity level based on factors such as age, sex, race, and pre-existing conditions.
This will help guide the hospitals in determining who to prioritize when there is a shortage of beds.
During the semester, we hope to be able to determine which conditions will leave somebody the
most vulnerable to severe complications or even death, and we hope hospitals can use this information to assign beds to those people,
and in the long run, save lives.

## **Methods**

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

### **Unsupervised Results**
Upon downloading our dataset, which was composed of patient-by-patient data describing things like sex, age, preexisting conditions, and symptoms, the first thing we had to do was make it suitable for machine learning methods in general. This means we had to eliminate columns/features that were extraneous or unrelated to our problem (such as factors which would be unknown at the time of arrival to the hospital and factors which were uniform over all patients). Then, we used pandas to convert our dataset into a dataframe, encoded categorical data into a one-hot format, and normalized data for use in a correlation map. Next, we moved on to performing key unsupervised learning techniques on our dataset, such as visualizations (correlation plots and heatmaps), dimensionality reduction (PCA), and clustering (K-means). These techniques provided insight into the structure of our data, what features correlated with others, what we could do to make supervised learning easier, and how the data clustered in its space.

<p align="center">
    <img src="assets/pairplot clean.png" width=50%/>
    <br>
    Pair Plots showing correlations and relationships between all numerical features and delimited by death/ICU.
    (click for more).
</p>

<p align="center">
    <img src="assets/pairplot detail.png" width=50%/>
    <br>
    Pair Plot detail
    <br>
    What these pair plots showed is not only the relationships of numerical factors with each other,
    but equally importantly, the univariate distributions of these factors split up based on class. 
    As shown, the distributions can help decide the relative importance of each factor by
    showing discrepancies between distributions for dead/seriously-affected patients and non-ICU patients.
</p>

<p align="center">
    <img src="assets/numcorr.png" width=50%/>
    <br>
    Correlation for numerical data
    <br>
    This visualization was important because it allowed us to see what factors had the most
    influence or correlation with the latent variables.
    Because certain factors are more important, we can cut off extraneous factors and create a simpler,
    faster, more understandable final model without having to record that many attributes of each patient.
</p>

<p align="center">
    <img src="assets/radviz.png" width=50%/>
    <br>
    RadViz for numerical data
    <br>
    Using our correlation data, we were able to depict a more graphical interpretation of what each factor means.
    This construction was built with scikit's Yellowbrick and shows a standardized view of how
    the factors most correlated with death/illness can be graphically distinguished from each other.
    From this graph, we see that entries with death/illness tend towards having higher levels of D-Dimer,
    which wasn't something that could be guessed without medical expertise.
</p>

## **Discussion**
Predicting risk based on demographic information, medical background, and behavior can provide extremely valuable insight
into how the COVID-19 pandemic should best be handled. At the institutional level, hospitals can use our risk predictions
to determine how to most efficiently allocate the limited resources in order to minimize deaths and complications.
Hospitals will be able to make well-informed, data-driven decisions for how to treat patients and what to be the most wary of.
Moreover, risk prediction and a strong understanding of what factors contribute the most to COVID-19 severity can also be informative for the individual.
An individual may engage in more extensive prevention behaviors if they are able to predict the severity of their illness or the illnesses of their loved ones.
Additionally, as a society, we can identify those individuals who are most at risk, and take extra precautions to protect them from the virus.
We hope that this increase in information will drive progress toward ending the pandemic.

### **Challenges: Unsupervised Portion**
One of our biggest challenges is that our dataset is very small and only has a little over 100 rows. We were unable to find larger datasets with de-identified patient data for free. Since we are interested in feature selection in order to determine which factors have the highest impact on COVID-19 outcomes, projecting our feature set onto a new basis using PCA may be unviable, so we will need to find an alternative method for dimensionality reduction that allows us to select the most important original features. We attempted to use a correlation matrix to identify the most heavily correlated features to eliminate, but the results were weaker than expected, potentially due to the one-hot encoded features. The one-hot encoding also made some of our initial methods, like a heatmap visualization, difficult to interpret.

### **Future Plans**
For our supervised portion, we're planning on using a Naive Bayes model to estimate a posterior probability of dying or having an illness based on attributes of each patient. What this will help us do is determine the severity of cases such that the most severe cases can have a larger priority when allocating ICU beds. Thus, the range of our values will be in the range [0,1] due to the nature of our posterior probability predictions, with probabilities skewed towards 0 because the chance of harm from COVID is considerable but generally low. This skew could be fixed by something like a sigmoid function, which would turn our Naive Bayes results into a more approachable probability distribution.

## **References**
- A. I. F. AI, “COVID-19 Open Research Dataset Challenge (CORD-19),” Kaggle, 28-Sep-2020. [Online]. Available: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558. [Accessed: 02-Oct-2020]. 
- “CDC COVID Data Tracker,” Centers for Disease Control and Prevention, 2020. [Online]. Available: https://covid.cdc.gov/covid-data-tracker/?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fcoronavirus%2F2019-ncov%2Fcases-updates%2Fcases-in-us.html. [Accessed: 02-Oct-2020]. 
- J. Turcotte, “Replication Data for: Risk Factors for Severe Illness in Hospitalized Covid-19 Patients at a Regional Hospital,” Harvard Dataverse, 22-Jul-2020. [Online]. Available: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FN2WZNK. [Accessed: 02-Oct-2020]. 
