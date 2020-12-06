#Deepak - PCA on Numerical data (not one-hot encoded data)
from sklearn.decomposition import PCA
df
#lets find all the columns with numerical (non-categorical data)
cols_with_numeric_data = ["age", "Ht", "Wgt", "BMI", "RRadmit", "HRadmit", "Systolic","Diastolic", "tempadmit", "O2admit", "OnsetDays", "Num_COVID_Symptoms", 
"Num_Other_Risk_Factor", "WBC", "Lympho", "Hg", "Plts", "AST", "Ddimer", "LDH", "CRP_Max_19k", "Ferrit", "Low02D1", "AvgMaxTemp", "LOS"]
cols_with_categorical_data = []
all_cols = list(df)
# print(cols_with_numeric_data)
# for i in cols_with_numeric_data:
#     print(df[i])
# print(all_cols)
for i in all_cols:
    if i not in cols_with_numeric_data:
        cols_with_categorical_data.append(i)
# print(cols_with_categorical_data)

numeric_df = df[cols_with_numeric_data]
categorical_df = df[cols_with_categorical_data]
numeric_df
#numeric df is df with only numeric columns
pca = PCA(n_components=5)
pca.fit(numeric_df)
#print(pca_df)
#pca.transform(numeric_df)
pca_df = pd.DataFrame(pca.components_) 
pca_df


#plot the variance
pca.n_components = 25
pca_data = pca.fit_transform(numeric_df)
percent_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percent_var_explained)

#plotting section
plt.figure(1, figsize = (6,4))
plt.clf()
plt.plot(cum_var_explained, linewidth = 2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

#Next
labels = list(numeric_df)
pca.n_components = 25
labeled_data = pca.fit_transform(numeric_df)
labeled_data = np.vstack((labeled_data.T, labels)).T
labeled_data



#Deepak PCA projection
from yellowbrick.features import ParallelCoordinates
from yellowbrick.datasets import load_occupancy


# Load the classification data set
#X, y = load_occupancy()
X = numericals[['Ddimer', 'WBC', 'age', 'LOS']]
y = numericals['Death_ICU']
#print(X[0].shape)
print(y[0].shape)
# Specify the features of interest and the classes of the target
features = ["Ddimer", "WBC", "age", "LOS"]
classes = ["alive", "dead"]

# Instantiate the visualizer
visualizer = ParallelCoordinates(classes=classes, features=features, sample=0.05, shuffle=True)

# Fit and transform the data to the visualizer
visualizer.fit_transform(X, y)

# Finalize the title and axes then display the visualization
visualizer.show()


#from yellowbrick.datasets import load_credit
# from yellowbrick.features import PCA as prinCompAnal

# # Specify the features of interest and the target
# #X, y = load_credit()
# #classes = ['account in default', 'current with bills']
# X = numericals
# y = numericals['Death_ICU']

# visualizer = PCA()
# visualizer.fit_transform(X, y)
# visualizer.show()