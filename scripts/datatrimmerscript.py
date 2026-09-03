import pandas as pd

# Input and output filenames
input_file = "nba-allplayers-careeraverage-data.csv"
output_file = "nba-allplayers-careeraverage-data-trimmed.csv"

# Read the CSV
df = pd.read_csv(input_file)

# Drop the first two columns
df = df.iloc[:, 2:]

# Save the result to a new CSV
df.to_csv(output_file, index=False)

print(f"Saved trimmed CSV as {output_file}")
