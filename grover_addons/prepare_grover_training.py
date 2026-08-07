import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare training data for Grover finetuning')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file with smiles and labels columns')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save prepared CSV file')
    return parser.parse_args()

def prepare_grover_csv(df, output_path, label_mapping={'active': 1, 'inactive': 0}):
    """
    Prepare CSV for Grover training
    
    Args:
        df: Input dataframe with 'smiles' and 'labels' columns
        output_path: Path to save the prepared CSV
        label_mapping: Mapping from string labels to numeric values
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Find smiles and labels columns (case-insensitive)
    smiles_col = None
    labels_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'smiles' in col_lower:
            smiles_col = col
        if 'label' in col_lower:
            labels_col = col
    
    if smiles_col is None or labels_col is None:
        raise ValueError(f"Could not find 'smiles' and 'labels' columns. Found columns: {df.columns.tolist()}")
    
    print(f"Using columns: smiles='{smiles_col}', labels='{labels_col}'")
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'smiles': df[smiles_col].str.strip()  # Strip whitespace from SMILES too
    })
    
    # Convert labels to numeric if ey are strings
    labels_data = df[labels_col]
    if labels_data.dtype == 'object' or labels_data.dtype == 'str':
        # Strip whitespace from labels
        labels_data = labels_data.str.strip()
        output_df['label'] = labels_data.map(label_mapping)
    else:
        output_df['label'] = labels_data
    
    # Check for any unmapped labels
    if output_df['label'].isna().any():
        unmapped = labels_data[output_df['label'].isna()].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}. Please update label_mapping.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} samples to {output_path}")
    print(f"Label distribution: {output_df['label'].value_counts().to_dict()}")

def main():
    args = parse_args()
    
    # Read input CSV
    print(f"Reading CSV file: {args.input_csv}")
    # sep=None sniffs the delimiter; both comma- and semicolon-separated splits occur.
    input_df = pd.read_csv(args.input_csv, sep=None, engine='python')
    
    print(f"Input shape: {input_df.shape}")
    print(f"Original columns: {input_df.columns.tolist()}")
    
    # Prepare for Grover
    prepare_grover_csv(input_df, args.output_csv)
    
    print(f"\nFile prepared successfully: {args.output_csv}")

if __name__ == '__main__':
    main()
