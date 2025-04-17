import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def load_data():
    crime_df = pd.read_csv('crime_data_2023.csv')
    weather_df = pd.read_csv('weather_data_2023.csv')
    return crime_df, weather_df


def prepare_data(crime_df, weather_df):
    # Rename columns for weather data
    weather_df.rename(columns={
        'time': 'date',
        'temperature_2m_max (°F)': 'temperature'
    }, inplace=True)

    # Convert dates
    crime_df['DATE OCC'] = pd.to_datetime(crime_df['DATE OCC'])
    crime_df['crime_date'] = crime_df['DATE OCC'].dt.date
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

    # Aggregate daily crime
    daily_crime = crime_df.groupby('crime_date').size().reset_index(name='crime_count')

    # Merge on date
    merged_df = pd.merge(daily_crime, weather_df, left_on='crime_date', right_on='date')
    merged_df.drop(columns=['date'], inplace=True)

    return merged_df


def visualize_data(df):
    sns.scatterplot(x='temperature', y='crime_count', data=df)
    plt.title('Temperature vs. Daily Crime Count (LA 2023)')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Crime Count')
    plt.tight_layout()
    plt.savefig('temp_vs_crime.png')
    plt.close()


def analyze_data(df):
    correlation = df['temperature'].corr(df['crime_count'])
    print(f'Correlation: {correlation:.3f}')

    X = sm.add_constant(df['temperature'])
    y = df['crime_count']
    model = sm.OLS(y, X).fit()

    # Print summary to terminal
    print(model.summary())

    # Write summary to text file
    with open('regression_summary.txt', 'w') as f:
        f.write(f'Correlation: {correlation:.3f}\n\n')
        f.write(model.summary().as_text())

    return model


def main():
    crime_df, weather_df = load_data()
    merged_df = prepare_data(crime_df, weather_df)
    visualize_data(merged_df)
    analyze_data(merged_df)


if __name__ == '__main__':
    main()
