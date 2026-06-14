"""
data/data_prep.py
Cleans and merges the two real datasets:
  1. 11_march_2025.csv  — personal transaction log (4597 rows, 33 months)
  2. banking-transaction-categorization-dataset.csv — banking records (200 rows)

Returns two clean DataFrames ready for ML pipelines:
  - df_personal  : date, category, amount, merchant, source
  - df_banking   : same schema, richer fields
  - df_combined  : merged for training
"""
import pandas as pd
import numpy as np
import re

# ── Category Normalisation Map 
# Maps raw personal-dataset spellings → clean canonical labels
CATEGORY_MAP = {
    "Restuarant":        "Food & Dining",
    "Coffe":             "Coffee & Cafes",
    "Market":            "Groceries",
    "Business lunch":    "Food & Dining",
    "Health":            "Health",
    "Clothing":          "Shopping",
    "Communal":          "Bills & Utilities",
    "Travel":            "Travel",
    "Learning":          "Education",
    "Events":            "Entertainment",
    "Tech":              "Shopping",
    "Sport":             "Health",
    "Other":             "Other",
    "Taxi":              "Transport",
    "Transport":         "Transport",
    "Phone":             "Bills & Utilities",
    "Motel":             "Travel",
    "joy":               "Entertainment",
    "business_expenses": "Business",
    "Fuel":              "Transport",
    "Rent Car":          "Transport",
    "Film/enjoyment":    "Entertainment",
    # Banking dataset categories (already clean)
    "Food & Drink":      "Food & Dining",
    "Groceries":         "Groceries",
    "Transportation":    "Transport",
    "Entertainment":     "Entertainment",
    "Utilities":         "Bills & Utilities",
    "Rent":              "Housing",
    "Home":              "Housing",
    "Insurance":         "Insurance",
    "Fees":              "Fees & Charges",
    "Travel":            "Travel",
    "Income":            "Income",
    "Salary":            "Income",
    "Transfer":          "Transfer",
    "Refund":            "Refund",
}

CANONICAL_CATEGORIES = sorted(set(CATEGORY_MAP.values()))


def load_personal(path: str) -> pd.DataFrame:
    """Load and clean 11_march_2025.csv / budget_data.csv / budjet__2_.csv"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date'])
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
    df = df.dropna(subset=['amount'])
    df = df[df['amount'] > 0]
    
    df['category_raw']  = df['category'].str.strip()
    df['category']      = df['category_raw'].map(CATEGORY_MAP).fillna('Other')
    df['merchant']      = df['category_raw']          # description is category in this dataset
    df['description']   = df['category_raw']
    df['is_recurring']  = False
    df['transaction_type'] = 'debit'
    df['source']        = 'personal'
    
    return df[['date','category','category_raw','merchant','description',
               'amount','is_recurring','transaction_type','source']]


def load_banking(path: str) -> pd.DataFrame:
    """Load and clean banking-transaction-categorization-dataset.csv"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df['date'] = pd.to_datetime(df['transaction_date'], dayfirst=True, errors='coerce').dt.tz_localize('UTC')
    df = df.dropna(subset=['date'])
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
    df = df.dropna(subset=['amount'])
    
    df['category_raw'] = df['category'].str.strip()
    df['category']     = df['category_raw'].map(CATEGORY_MAP).fillna('Other')
    df['merchant']     = df['merchant_name'].fillna(df['transaction_description'].str[:30])
    df['description']  = df['transaction_description'].fillna(df['merchant_name'])
    df['is_recurring'] = df['is_recurring'].fillna(False).astype(bool)
    df['transaction_type'] = df.get('transaction_type', 'debit')
    df['source']       = 'banking'
    
    return df[['date','category','category_raw','merchant','description',
               'amount','is_recurring','transaction_type','source']]


def build_combined(personal_path: str, banking_path: str) -> dict:
    """
    Returns a dict with keys:
      personal, banking, combined, categorization_train
    """
    df_p = load_personal(personal_path)
    df_b = load_banking(banking_path)
    df_all = pd.concat([df_p, df_b], ignore_index=True).sort_values('date')
    
    # Build categorization training set — (description, category) pairs
    # Use banking dataset (richer descriptions) + personal
    train = df_all[['description','category']].dropna()
    train = train[train['description'].str.strip() != '']
    train = train[train['category'] != 'Other']   # keep only confident labels
    
    return {
        'personal':             df_p,
        'banking':              df_b,
        'combined':             df_all,
        'categorization_train': train,
    }


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined data into monthly category totals."""
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    expenses = df[df['transaction_type'] != 'credit']
    return (
        expenses.groupby(['month', 'category'])['amount']
        .sum()
        .reset_index()
        .rename(columns={'amount': 'total'})
    )


if __name__ == "__main__":
    data = build_combined(
        "dataset/11 march 2025.csv",
        "dataset/banking-transaction-categorization-dataset.csv",
    )
    print(f"Personal:  {len(data['personal'])} rows")
    print(f"Banking:   {len(data['banking'])} rows")
    print(f"Combined:  {len(data['combined'])} rows")
    print(f"Train set: {len(data['categorization_train'])} rows")
    print(f"\nCanonical categories: {CANONICAL_CATEGORIES}")
    print(f"\nCategory distribution:\n{data['combined']['category'].value_counts().to_string()}")
