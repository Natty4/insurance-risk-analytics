# Import required libraries
import pandas as pd
import numpy as np

# Define the title-to-gender mapping
TITLE_TO_GENDER = {
    'Mr': 'Male',
    'Mrs': 'Female',
    'Ms': 'Female',
    'Miss': 'Female',
    'Dr': 'Not specified',  # You can extend this as needed
}

# Function to standardize missing values
def standardize_missing_values(df, cols):
    """
    Standardizes common missing value placeholders like 'NA', 'Unknown', etc. to NaN.
    """
    for col in cols:
        df[col] = df[col].replace(['', 'Not specified', 'Unknown', 'NA', 'N/A'], np.nan)
    return df

# Function to infer gender from title
def infer_gender_from_title(row):
    """
    Infers the gender based on the Title, if Gender is missing.
    """
    if pd.isna(row['Gender']):
        return TITLE_TO_GENDER.get(row['Title'], np.nan)
    return row['Gender']

# Function to impute gender using the title
def impute_gender(df):
    """
    Imputes missing 'Gender' values based on the 'Title' column.
    """
    df['Gender'] = df.apply(infer_gender_from_title, axis=1)
    return df

# Function to impute categorical variables using mode
def impute_categorical_modes(df, columns):
    """
    Imputes missing values for categorical variables using the mode (most frequent value).
    """
    for col in columns:
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)
    return df

# Function to clean the data by standardizing missing values, imputing gender and categorical modes
def clean_data(df):
    """
    The main function to clean the data: handles missing values and imputations.
    """
    # Standardize placeholders to NaN
    df = standardize_missing_values(df, ['Gender', 'Bank', 'AccountType'])
    
    # Impute gender using Title
    df = impute_gender(df)
    
    # Impute other categorical variables like 'Bank' and 'AccountType'
    df = impute_categorical_modes(df, ['Bank', 'AccountType'])
    
    return df

# Function to safely drop unnecessary columns
def drop_columns_safely(df, cols_to_drop):
    """
    Drops specified columns from the dataframe, checks if they exist before dropping.
    """
    dropped_cols = []
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            dropped_cols.append(col)
    print(f"Dropped columns: {dropped_cols}")
    return df

# Function to impute missing 'CustomValueEstimate' using median by 'make'
def impute_custom_value(df):
    """
    Imputes missing 'CustomValueEstimate' values using group median by 'make' and global median as fallback.
    """
    # Step 1: Median by VehicleMake
    make_median = df.groupby('make')['CustomValueEstimate'].transform('median')

    # Step 2: Impute missing values with group median
    df['CustomValueEstimate_imputed'] = df['CustomValueEstimate'].fillna(make_median)

    # Step 3: Fallback to global median for still missing values
    global_median = df['CustomValueEstimate'].median()
    df['CustomValueEstimate_imputed'] = df['CustomValueEstimate_imputed'].fillna(global_median)

    # Replace original column with imputed values
    df['CustomValueEstimate'] = df['CustomValueEstimate_imputed']
    df.drop(columns=['CustomValueEstimate_imputed'], inplace=True)

    return df

# Function to calculate Loss Ratio and handle zero premiums
def calculate_loss_ratio(df):
    """
    Calculates the loss ratio while handling zero premiums gracefully.
    When TotalPremium == 0, set LossRatio to 100% (i.e., 1).
    """
    df['LossRatio'] = df.apply(
        lambda row: 1 if row['TotalPremium'] == 0 else row['TotalClaims'] / row['TotalPremium'],
        axis=1
    )
    return df

# Main function to clean and process the data
def preprocess_data(input_file, output_file):
    """
    Full preprocessing pipeline including cleaning, imputations, and saving the cleaned data.
    """
    try:
        # Read the raw data
        df = pd.read_csv(input_file, sep="|")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    
    # Preview Before Cleaning
    print("🔍 BEFORE CLEANING")
    print("\nGENDER (missing count):", df['Gender'].isna().sum())
    print(df['Gender'].value_counts(dropna=False))
    print("\nBANK (missing count):", df['Bank'].isna().sum())
    print(df['Bank'].value_counts(dropna=False).head())
    print("\nACCOUNT TYPE (missing count):", df['AccountType'].isna().sum())
    print(df['AccountType'].value_counts(dropna=False).head())

    # Apply Cleaning Process
    df_cleaned = clean_data(df.copy())

    # Drop columns that are not needed
    columns_to_drop = [
        'UnderwrittenCoverID', 'PolicyID', 'Language', 'Country',
        'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet', 'Title'
    ]
    df_cleaned = drop_columns_safely(df_cleaned, columns_to_drop)

    # Impute 'CustomValueEstimate' based on the group and global median
    df_cleaned = impute_custom_value(df_cleaned)

    # Calculate Loss Ratio
    df_cleaned = calculate_loss_ratio(df_cleaned)

    # Impute 'WrittenOff' using the mode
    mode_writtenoff = df_cleaned['WrittenOff'].mode()[0]
    df_cleaned['WrittenOff'] = df_cleaned['WrittenOff'].fillna(mode_writtenoff)

    # Save the cleaned data to CSV
    try:
        df_cleaned.to_csv(output_file, index=False)
        print("✅ Final cleaned data saved.")
    except Exception as e:
        print(f"Error while saving the file: {e}")
        return None
    
    # Return cleaned dataframe for further analysis
    return df_cleaned