import pandas as pd
import ast

def validate_dataset(csv_path="ground_truth.csv"):
    print(f"Validating '{csv_path}'...\n")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    required_columns = ["question", "ground_truth", "contexts"]
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Missing required column: '{col}'")
            return

    print(f"✅ Total test cases loaded: {len(df)}")
    
    if len(df) < 15:
        print(f"⚠️ Warning: Found {len(df)} cases. Aim for at least 15 cases for statistically meaningful evaluation.")
    else:
        print("✅ Dataset size meets recommended baseline (>= 15 test cases).")

    # Inspect context list parsing
    sample_context = df.iloc[0]["contexts"]
    try:
        parsed_context = ast.literal_eval(sample_context) if isinstance(sample_context, str) else sample_context
        if isinstance(parsed_context, list):
            print("✅ Context column correctly formatted as a list of strings.")
        else:
            print("⚠️ Context column parsed, but is not formatted as a list.")
    except Exception as e:
        print(f"⚠️ Context string parsing notice: {e}")

    print("\n--- First Test Case Preview ---")
    print(f"Q: {df.iloc[0]['question']}")
    print(f"A: {df.iloc[0]['ground_truth']}")
    print(f"C: {df.iloc[0]['contexts']}")

if __name__ == "__main__":
    validate_dataset()