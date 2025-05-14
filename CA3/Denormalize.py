import pandas as pd
from datetime import timedelta

train_df = pd.read_csv("regression-dataset-train.csv")
test_df = pd.read_csv("regression-dataset-test-unlabeled.csv")
external_df = pd.read_csv(r"content\additional_data.csv")
external_df.columns = external_df.columns.str.strip()

external_df = external_df.drop(columns=['casual', 'registered'])

train_df['date'] = pd.to_datetime(train_df['date'], dayfirst=True, errors='coerce')
test_df['date'] = pd.to_datetime(test_df['date'], dayfirst=True, errors='coerce')
external_df['date'] = pd.to_datetime(external_df['date'], format='%Y-%m-%d', errors='coerce')

external_df['date'] = external_df['date'] + pd.DateOffset(years=7)

raw_df = pd.concat([train_df, test_df], ignore_index=True)
raw_df.set_index('date', inplace=True)

numerical_cols = ['temperature', 'feels_like_temp', 'humidity', 'wind_speed']

missing = 0
for i in range(len(external_df)):
    actual_date = external_df.loc[external_df.index[i], 'date']
    if actual_date in raw_df.index:
        for col in numerical_cols:
            external_df.loc[external_df.index[i], col] = raw_df.loc[actual_date, col]
    else:
        missing += 1
        print(f"No match for date {actual_date}")

external_df.to_csv("corrected_external_data.csv", index=False)
print("\n Saved as corrected_external_data.csv with updated date column")

if missing:
    print(f"Total unmatched rows: {missing}")