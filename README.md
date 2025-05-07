import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('ecommerce_returns.csv')

# Return rate by category
category_returns = df.groupby('Product Category')['Returned (Y/N)'].apply(lambda x: (x == 'Y').mean() * 100)
category_returns.plot(kind='bar', title='Return Rate by Category')
plt.ylabel('Return Rate (%)')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[['Price', 'Delivery Days']]  # Add more features
y = (df['Returned (Y/N)'] == 'Y').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
