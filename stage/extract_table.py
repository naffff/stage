from tabula import read_pdf
import pandas as pd

# Read the PDF and extract tables
tables = read_pdf("lpi.pdf", pages='all', multiple_tables=True)

# Save each table as a separate CSV file
for i, table in enumerate(tables):
    # Convert the table to a DataFrame
    df = pd.DataFrame(table)

    # Generate the output file name
    output_file = f"output_table_result_{i+1}.csv"

    # Save the DataFrame as a CSV file
    df.to_csv(output_file, index=False)

    print(f"Table {i+1} saved as {output_file}")
