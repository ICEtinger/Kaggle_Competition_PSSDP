import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def gini(actual, pred, cmpcol = 0, sortcol = 1):
      assert( len(actual) == len(pred) )
      all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
      all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
      totalLosses = all[:,0].sum()
      giniSum = all[:,0].cumsum().sum() / totalLosses
      
      giniSum -= (len(actual) + 1) / 2.
      return giniSum / len(actual)

def gini_normalized(a, p):
      return gini(a, p) / gini(a, a)

def test_gini():
      def fequ(a,b):
            return abs( a -b) < 1e-6
      def T(a, p, g, n):
            assert( fequ(gini(a,p), g) )
            assert( fequ(gini_normalized(a,p), n) )
      T([1, 2, 3], [10, 20, 30], 0.111111, 1)
      T([1, 2, 3], [30, 20, 10], -0.111111, -1)
      T([1, 2, 3], [0, 0, 0], -0.111111, -1)
      T([3, 2, 1], [0, 0, 0], 0.111111, 1)
      T([1, 2, 4, 3], [0, 0, 0, 0], -0.1, -0.8)
      T([2, 1, 4, 3], [0, 0, 2, 1], 0.125, 1)
      T([0, 20, 40, 0, 10], [40, 40, 10, 5, 5], 0, 0)
      T([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5], 0.171428, 0.6)
      T([40, 20, 10, 0, 0], [40, 20, 10, 0, 0], 0.285714, 1)
      T([1, 1, 0, 1], [0.86, 0.26, 0.52, 0.32], -0.041666, -0.333333)

# Create an object (dataframe) called df
# each line in the file is a data point. The first columns are the features. 
# The last column is the value we are interested at.
df = pd.read_csv('train.csv')

#df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
#train, test = df[df['is_train']==True], df[df['is_train']==False]

#print('Number of observations in the training data:', len(train))
#print('Number of observations in the test data:',len(test))


# "features" is df without the first two columns (i.e.: the features).
features = df[df.columns[2:]]

# "y" is the second column of df (i.e: the target of each data point).
y = df[df.columns[1:2]]
print("> > > The y values given are:\n",
      y)

# Create a random forest classifier.
clf = RandomForestRegressor(n_jobs=1)

# Train the classifier to take the training features and learn how they relate
# to the training classification
clf.fit(features, y)

predicted = clf.predict(features)
print("> > > The y values predicted by the Random Forest Regression are:\n", 
      predicted)

print("> > > The normalized gini coefficient is:\n", 
      gini_normalized(y, predicted))


# Create a random forest classifier.
#clf2 = MLPRegressor(alpha=10)

# Train the classifier to take the training features and learn how they relate
# to the training classification
#clf2.fit(features, y)

#print("> > > The y values predicted by the Neural Net Regression are:\n",
#      clf2.predict(features))
