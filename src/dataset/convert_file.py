import pandas as pd
import os


def convert_file(input_path, output_format):
    # Supported output formats
    supported_formats = ['csv', 'parquet', 'pkl']

    # Validate the output format
    if output_format not in supported_formats:
        raise ValueError(
            f"Unsupported output file format. Supported formats are: {', '.join(supported_formats)}")

    # Determine the input format based on the file extension
    input_extension = os.path.splitext(input_path)[1].lower()

    # Read the input file
    if input_extension == '.csv':
        df = pd.read_csv(input_path)
    elif input_extension == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_extension == '.pkl':
        df = pd.read_pickle(input_path)
    else:
        raise ValueError("Unsupported input file format")

    # Determine the output path
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"

    # Save the file in the desired format
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif output_format == 'pkl':
        df.to_pickle(output_path)

    print(f"File saved as {output_path}")
