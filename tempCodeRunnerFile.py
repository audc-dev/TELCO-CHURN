import os
print("Current directory:", os.getcwd())
for root, dirs, files in os.walk('.'):
    if 'telco_churn.csv' in files:
        print(f"File found at: {os.path.join(root, 'telco_churn.csv')}")