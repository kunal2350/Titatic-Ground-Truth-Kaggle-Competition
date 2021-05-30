# Titatic-Ground-Truth-Kaggle-Competition

<p>This dataset has been taken from <a href="https://www.kaggle.com/c/titanic">https://www.kaggle.com/c/titanic</a>. The adjoined task along with this dataset is to predict the survival of the passengers whose details are listed in the test.csv file. This project is a comparative classification model for accomplishing this task.</p>

<h5>The overall algorithm used for prediction of survival of the passengers is as follows:</h5>
<ol> 
    <li>Import both the datasets (train and test) and all the necessary libraries required for processing and prediction of data.</li>
    <li>Do some EDA to get to know some crucial informatiion about the dataset like the number of attributes, types of attributes (categorial, metric, etc), datatypes of the values stored in the attributes, and the nulls and duplicates in the dataset.</li>
    <li>Use logic to eliminate the attributes clearly not necessary to accomplish the task of prediction i.e. they wouldn't have played any role in the survival or decease of the subjects mentions in the dataset.</li>
    <li>Remove the null values from the dataset according to the attribute and its values.</li>
    <li>Identify the categorical attributes and try to assign them ordinal integer values for better prediction results</li>
    <li>Once this whole EDA and preprocessing of data is complete, divide the data into training sets (X_train, Y_train) and test set (X_test).</li>
    <li>Fit the training sets to the different machine learning algorithms models and then pass the X_test through those to check the accuracy of each and every model.</li>
    <li>Choose the most accurate and efficient model for any further application with the related data and task.</li>
</ol>

<h5>The Machine Learning algorithms used in this project are:</h5>
<ol>
    <li>Logistic Regression</li>
    <li>Support Vector Machine</li>
    <li>KNN</li>
    <li>Naive Bayes</li>
    <li>Decision Tree</li>
    <li>Random Forest</li>

<h2>Importing the libraries</h2>


```python
import pandas as pd
import numpy as np
import random as rnd
```


```python
import dython 
from dython.nominal import associations
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
```

<h2>Importing the datasets</h2>


```python
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
combine = [train_set, test_set]
```

<h2>Exploratory Data Analysis of the imported data</h2>


```python
train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set.info()
print('_'*40)
test_set.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    ________________________________________
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    


```python
train_set.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set.describe(include=['O'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Meyer, Mr. August</td>
      <td>male</td>
      <td>CA. 2343</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>



<ul><b>From the above EDA, few new crucialthings can be inferred are:</b>
    <li>Cabin values have several dupicates across samples. </li>
    <li>Embarked takes three possible values. S port used by most passengers.</li>
    <li>Ticket feature has high ratio (22%) of duplicate values (unique=681).</li>
    </ul>

<ul> <b>Some changes and improvements can be made to the working dataset in order to make the prediction more precise.</b>
    <li>PassengerId and Name can be dropped from the working dataset as they don't have any correlation with the probability of a person surviving in the practical scenario.</li>
    <li>The cabin feature can be dropped as well because it has a very high nuber of null values in both training and testing data.</li>
    <li>The ticket column can be dropped as it is also contains a lot of null values as well as it won't have any correlation with the probability of surviving in the practical scenario.</li>
    <li>The nulls in the age feature should be replaced with a calculated guess for the age of the passenger.</li>
    <li>The values in the sex and embarked attribute can be replaced with ordinal integer values for convenience.</li>
    </ul>

<h2> Dropping the logically uncorrelated parameters from the dataset</h2>


```python
train_set = train_set.drop(['Ticket', 'Cabin', 'Name','PassengerId' ], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1)
combine = [train_set, test_set]
train_set
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



<h2>Replacing the nulls and assigning ordinal integers to the categorical features</h2>


```python
guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

freq_port = train_set.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
test_set['Fare'].fillna(test_set['Fare'].dropna().median(), inplace=True)
```

<p>The null values in the age and embarked columns of the training set and fare column of the test set have been replaced.</p>


```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
for df in combine:
    df[["Sex","Embarked"]] = enc.fit_transform(df[["Sex","Embarked"]])
```


```python
train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>13.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>30.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>1.0</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>23.45</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>30.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>7.75</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



<p>The categorical values male and female in the attribute 'Sex' have been ordinally encoded to 0.0 and 1.0 respectively. Similarly, the categorical values S, C and Q in the attribute 'Embarked' have been ordinally encoded to 0.0, 1.0, 2.0 respectively.</p>

<h2>Visualizing the correlation of the parameters with survival probability</h2>


```python
cols = train_set.columns
associations(train_set[cols],figsize=(10,10))
```


![png](output_27_0.png)





    {'corr':           Survived    Pclass       Sex       Age     SibSp     Parch  \
     Survived  1.000000 -0.338481  0.543351 -0.060291 -0.035322  0.081629   
     Pclass   -0.338481  1.000000 -0.131900 -0.414682  0.083081  0.018443   
     Sex       0.543351 -0.131900  1.000000 -0.104584  0.114631  0.245489   
     Age      -0.060291 -0.414682 -0.104584  1.000000 -0.250248 -0.175708   
     SibSp    -0.035322  0.083081  0.114631 -0.250248  1.000000  0.414838   
     Parch     0.081629  0.018443  0.245489 -0.175708  0.414838  1.000000   
     Fare      0.257307 -0.549500  0.182333  0.123250  0.159651  0.216225   
     Embarked  0.106811  0.045702  0.116569 -0.059221 -0.059961 -0.078665   
     
                   Fare  Embarked  
     Survived  0.257307  0.106811  
     Pclass   -0.549500  0.045702  
     Sex       0.182333  0.116569  
     Age       0.123250 -0.059221  
     SibSp     0.159651 -0.059961  
     Parch     0.216225 -0.078665  
     Fare      1.000000  0.062142  
     Embarked  0.062142  1.000000  ,
     'ax': <matplotlib.axes._subplots.AxesSubplot at 0x20f8a5c9550>}



<h2>Splitting of dataset</h2>


```python
X_train = train_set.drop("Survived", axis=1)
Y_train = train_set["Survived"]
X_test  = test_set
X_train.shape, Y_train.shape, X_test.shape
```




    ((891, 7), (891,), (418, 7))




```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.0</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1.0</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<p>The "Survived" column has been dropped from the train_set to make the X_train dataset to be used for training the models.<p>


```python
Y_train.head()
```




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: Survived, dtype: int64



<p>The values of the "Survived" column has been stored in the Y_train dataset. The values match the index of the X_train dataset.<p>


```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1.0</td>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1.0</td>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<h2>Training of differnet machine learning models</h2>


```python
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```




    68.35




```python
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```




    97.98




```python
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```




    97.98




```python
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```




    79.24




```python
#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```




    84.4




```python
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
```

    D:\Anaconda3\envs\tf\lib\site-packages\sklearn\linear_model\_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    81.03



<h2>Comparing the accuracies of the different models trained</h2>


```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_decision_tree]})
models = models.sort_values(by='Score', ascending=False)
models
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>97.98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Decision Tree</td>
      <td>97.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>84.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression</td>
      <td>81.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>79.24</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Support Vector Machines</td>
      <td>68.35</td>
    </tr>
  </tbody>
</table>
</div>



<p>It is clear from the above table that Random Forest and Decision Tree are the best algorithms for the prediction of survival of passenger with an accuracy of 97.98%. On the other hand, Support Vector Machine performed the worst with just 68.35% accuracy.</p>

<h2>Confusion Matrix and classification Report</h2>


```python
from sklearn.metrics import confusion_matrix

y_pred_train = decision_tree.predict(X_train)

cm = confusion_matrix(Y_train, y_pred_train)
print('Confusion matrix\n\n', cm)

from sklearn.metrics import classification_report
print(classification_report(Y_train, y_pred_train))

```

    Confusion matrix
    
     [[546   3]
     [ 15 327]]
                  precision    recall  f1-score   support
    
               0       0.97      0.99      0.98       549
               1       0.99      0.96      0.97       342
    
        accuracy                           0.98       891
       macro avg       0.98      0.98      0.98       891
    weighted avg       0.98      0.98      0.98       891
    
    

<h2>Prediction of given test set using Decision Tree Algorithm</h2>


```python
Y_pred = decision_tree.predict(X_test)
Y_pred
```




    array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
           0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
           1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
          dtype=int64)



<h2>Conclusion</h2>

<p>The survival of the passengers whose data has been provided in the test.csv dataset has been predicted with <b>97.98% accuracy</b> using the <b>Decision Tree Classification Model</b>. Alternatively, it can also be done with the Random Forest Classification Model as it has the same accuracy performance.</p>

<p> Project by - <b>KUNAL NAG</b></p>


```python

```
