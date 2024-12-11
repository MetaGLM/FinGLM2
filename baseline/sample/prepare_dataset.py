import pandas as pd
import argparse


def excel_to_csv(excel_file_path, csv_file_path):
    """
    Read an Excel file and save it as a CSV file.
    :param excel_file_path: Path to the Excel file
    :param csv_file_path: Path to save the CSV file
    """

    data = pd.read_excel(excel_file_path)
    data.to_csv(csv_file_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_file", type=str, required=True, help="Path to the Excel file")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to save the CSV file")
    args = parser.parse_args()
    excel_to_csv(args.excel_file, args.csv_file)
