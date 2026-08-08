import pandas as pd
import argparse
import os
import csv

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def read_input_table(input_path):
    if input_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(input_path)

    try:
        # sep=None sniffs the delimiter. A semicolon-separated file does not raise
        # ParserError -- it parses into a single column -- so the fallback below
        # would never fire and the malformed frame would pass through silently.
        return pd.read_csv(input_path, sep=None, engine='python')
    except pd.errors.ParserError:
        rows = []
        with open(input_path, 'r', encoding='utf-8') as handle:
            reader = csv.reader(handle, skipinitialspace=True)
            try:
                header_parts = [part.strip() for part in next(reader)]
            except StopIteration:
                raise ValueError(f"Input file is empty: {input_path}")

            if len(header_parts) < 4:
                raise ValueError(
                    f"Expected at least 4 columns in header for fallback parsing, found: {header_parts}"
                )

            for parts in reader:
                if not parts or not any(str(cell).strip() for cell in parts):
                    continue

                if len(parts) < 4:
                    continue

                if len(parts) > 4:
                    parts = [parts[0], parts[1], parts[2], ','.join(parts[3:])]

                rows.append({
                    header_parts[0]: str(parts[0]).strip(),
                    header_parts[1]: str(parts[1]).strip(),
                    header_parts[2]: str(parts[2]).strip().strip('"'),
                    header_parts[3]: str(parts[3]).strip()
                })

        return pd.DataFrame(rows)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare CSV for Grover feature extraction')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input file (CSV or Excel)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to output CSV file')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='Name of SMILES column')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Name of label column (if exists and is numeric)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read input file
    print(f"Reading input file: {args.input_csv}")
    df = read_input_table(args.input_csv)
        
    print(f"Input shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    smiles_column = args.smiles_column
    if smiles_column not in df.columns:
        lowered = {str(col).strip().lower(): col for col in df.columns}
        smiles_column = lowered.get(smiles_column.lower())
    if smiles_column is None or smiles_column not in df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in input file. Available columns: {df.columns.tolist()}")
    
    # Create output dataframe with SMILES and dummy label
    output_df = pd.DataFrame({
        'smiles': df[smiles_column],
        'label': 0  # Dummy label for feature extraction
    })
    
    # Remove any rows with missing SMILES
    output_df = output_df.dropna(subset=['smiles'])

    # Normalize and optionally validate SMILES
    output_df['smiles'] = output_df['smiles'].astype(str).str.strip()
    output_df = output_df[output_df['smiles'] != '']

    if Chem is not None:
        valid_mask = output_df['smiles'].apply(lambda smi: Chem.MolFromSmiles(smi) is not None)
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            print(f"Dropping {dropped} invalid SMILES rows.")
        output_df = output_df[valid_mask]
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save output
    output_df.to_csv(args.output_csv, index=False)
    print(f"Prepared CSV saved to: {args.output_csv}")
    print(f"Output shape: {output_df.shape}")
    print(f"Number of molecules: {len(output_df)}")
    print(f"\nFirst few rows:")
    print(output_df.head())

if __name__ == '__main__':
    main()
