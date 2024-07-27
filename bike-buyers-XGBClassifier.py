import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('./data/bike_buyers.csv')
print(df.head())

# null value
print(df.isnull().sum())

# gender
p_male = df[df['Gender'] == 'Male']['Purchased Bike'].value_counts()
p_female = df[df['Gender'] == 'Female']['Purchased Bike'].value_counts()
ds = pd.DataFrame([p_male, p_female])
ds.index = ['Male', 'Female']
ds.plot(kind='bar', figsize=(6, 4), color=['gold', 'dodgerblue'],
        label=['Not Purchased Bike', 'Purchased Bike'])
plt.show()

# age
figure = plt.figure(figsize=(15, 8))
plt.hist([df[df['Purchased Bike'] == 'No']['Age'],
          df[df['Purchased Bike'] == 'Yes']['Age']],
         color=['gold', 'dodgerblue'],
         label=['Not Purchased Bike', 'Purchased Bike'])
plt.xlabel('Age')
plt.ylabel('Population')
plt.legend()
plt.show()

# income
figure = plt.figure(figsize=(15, 8))
plt.hist([df[df['Purchased Bike'] == 'No']['Income'],
          df[df['Purchased Bike'] == 'Yes']['Income']],
         color=['gold', 'dodgerblue'],
         label=['Not Purchased Bike', 'Purchased Bike'])
plt.xlabel('Income')
plt.ylabel('Population')
plt.legend()
plt.show()

# occupation
purchased_counts = df[df['Purchased Bike']
                      == 'Yes']['Occupation'].value_counts()
not_purchased_counts = df[df['Purchased Bike']
                          == 'No']['Occupation'].value_counts()

occupations = df['Occupation'].unique()
purchased_counts = purchased_counts.reindex(occupations, fill_value=0)
not_purchased_counts = not_purchased_counts.reindex(occupations, fill_value=0)

figure = plt.figure(figsize=(12, 6))
x = range(len(occupations))
bar_width = 0.4

plt.bar(x, not_purchased_counts, width=bar_width,
        color='gold', label='Not Purchased Bike')
plt.bar([p + bar_width for p in x], purchased_counts,
        width=bar_width, color='dodgerblue', label='Purchased Bike')
plt.xticks([p + bar_width / 2 for p in x], occupations, rotation=90)

plt.xlabel('Occupation')
plt.ylabel('Population')
plt.legend()
plt.show()

# commute distance
purchased_counts = df[df['Purchased Bike'] ==
                      'Yes']['Commute Distance'].value_counts()
not_purchased_counts = df[df['Purchased Bike']
                          == 'No']['Commute Distance'].value_counts()
desired_order = ['0-1 Miles', '1-2 Miles',
                 '2-5 Miles', '5-10 Miles', '10+ Miles']

purchased_counts = purchased_counts.reindex(desired_order, fill_value=0)
not_purchased_counts = not_purchased_counts.reindex(
    desired_order, fill_value=0)

figure = plt.figure(figsize=(12, 6))
x = range(len(desired_order))
bar_width = 0.4

plt.bar(x, not_purchased_counts, width=bar_width,
        color='gold', label='Not Purchased Bike')
plt.bar([p + bar_width for p in x], purchased_counts,
        width=bar_width, color='dodgerblue', label='Purchased Bike')
plt.xticks([p + bar_width / 2 for p in x], desired_order, rotation=90)

plt.xlabel('Commute Distance')
plt.ylabel('Population')
plt.legend()

plt.show()

# region
purchased_counts = df[df['Purchased Bike'] == 'Yes']['Region'].value_counts()
not_purchased_counts = df[df['Purchased Bike']
                          == 'No']['Region'].value_counts()
regions = df['Region'].unique()

purchased_counts = purchased_counts.reindex(regions, fill_value=0)
not_purchased_counts = not_purchased_counts.reindex(regions, fill_value=0)

figure = plt.figure(figsize=(12, 6))
x = range(len(regions))
bar_width = 0.4

plt.bar(x, not_purchased_counts, width=bar_width,
        color='gold', label='Not Purchased Bike')
plt.bar([p + bar_width for p in x], purchased_counts,
        width=bar_width, color='dodgerblue', label='Purchased Bike')
plt.xticks([p + bar_width / 2 for p in x], regions, rotation=90)

plt.xlabel('Region')
plt.ylabel('Population')
plt.legend()
plt.show()

# fillna
marital_status_mode = df['Marital Status'].mode()[0]
df['Marital Status'] = df['Marital Status'].fillna(marital_status_mode)

gender_mode = df['Gender'].mode()[0]
df['Gender'] = df['Gender'].fillna(gender_mode)

mean_income = df['Income'].mean(skipna=True)
df["Income"] = df["Income"].fillna(mean_income)

mean_children = df['Children'].mean(skipna=True)
df['Children'] = df['Children'].fillna(mean_children)

home_owne_mode = df['Home Owner'].mode()[0]
df['Home Owner'] = df['Home Owner'].fillna(home_owne_mode)

mean_cars = df['Cars'].mean(skipna=True)
df['Cars'] = df['Cars'].fillna(mean_cars)

mean_age = df['Age'].mean(skipna=True)
df["Age"] = df["Age"].fillna(mean_age)

# label encoding and get dummies
# Marital Status: 'Married'=1, 'Single'=0
df['Marital Status'] = df['Marital Status'].map({'Married': 1, 'Single': 0})

# Gender: 'Female'=1, 'Male'=0
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Education: 'Partial High School'=1, 'High School'=2, 'Partial College'=3, 'Bachelors'=4, 'Graduate Degree'=5
education_order = {
    'Partial High School': 1,
    'High School': 2,
    'Partial College': 3,
    'Bachelors': 4,
    'Graduate Degree': 5
}
df['Education'] = df['Education'].map(education_order)

# Occupation: get_dummies
df = pd.get_dummies(df, columns=['Occupation'], prefix='Occupation')

# Home Owner: 'Yes'=1, 'No'=0
df['Home Owner'] = df['Home Owner'].map({'Yes': 1, 'No': 0})

# Commute Distance: '0-1 Miles'=1, '1-2 Miles'=2, '2-5 Miles'=3, '5-10 Miles'=4, '10+ Miles'=5
commute_order = {
    '0-1 Miles': 0.5,
    '1-2 Miles': 1.5,
    '2-5 Miles': 3.5,
    '5-10 Miles': 7.5,
    '10+ Miles': 10
}
df['Commute Distance'] = df['Commute Distance'].map(commute_order)

# Region: get_dummies
df = pd.get_dummies(df, columns=['Region'], prefix='Region')

# Purchased Bike: 'Yes'=1, 'No'=0
df['Purchased Bike'] = df['Purchased Bike'].map({'Yes': 1, 'No': 0})

# convert column to float
for col in df.columns:
    df[col] = df[col].astype(float)

# correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(24, 20))

cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=.5)
plt.title('Correlation Matrix')

plt.show()

# split train and test, standard scaler
df = df.drop(columns=['ID'])
X = df.drop(columns=['Purchased Bike'])
y = df['Purchased Bike']

scaler = StandardScaler()
X_scl = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scl, y, test_size=0.2, random_state=0)

# model
model = XGBClassifier(n_estimators=90, max_depth=9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3, 2.3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            0, 1], yticklabels=[0, 1], annot_kws={'size': 20})
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
