#Implementarea unui program capabil să citească setul de date și să semnaleze eventuale erori
# (valori lipsă sau suplimentare, instanțe identice).

import pandas as pd

file_path = "C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx"
xls = pd.ExcelFile(file_path)

data_df = pd.read_excel(xls, sheet_name='Data')
code_df = pd.read_excel(xls, sheet_name='Code')

validation_dict = {}


for index, row in code_df.iterrows():
    variable = row['Variable']
    if pd.notnull(row['Values']):
        validation_dict[variable] = set(row['Values'].split('/'))
print(validation_dict)


def check_columns_for_errors():
    erori = {}
    for column in data_df.columns:
        if column in validation_dict:
            valori_asteptate = validation_dict[column]
            for element in data_df[column]:
                if str(element) not in valori_asteptate:
                    if column not in erori:
                        erori[column] = {}
                    if element not in erori[column]:
                        erori[column][element] = 0
                    erori[column][element] += 1

    return erori


def check_columns_for_missing_values():
    missing_values = data_df.isnull().sum()
    # print(missing_values)

    print("\nRaport valori lipsa:")
    if missing_values.any():
        print("\nSunt valori lipsă în următoarele coloane:")
        print(missing_values[missing_values > 0])
    else:
        print("Nu sunt valori lipsă în setul de date.")


def check_columns_for_duplicates():
    subset_columns = data_df.columns[2:]

    duplicate_rows = data_df[data_df.duplicated(subset=subset_columns, keep=False)]
    if not duplicate_rows.empty:
        print(f"\nAu fost găsite {duplicate_rows.shape[0]} duplicate:")
        print(duplicate_rows)
    else:
        print("\nNu au fost găsite duplicate.")


def genereaza_raport_erori(erori):
    print("Raport de erori:")
    if not erori:
        print("Nu au fost găsite erori.")
    else:
        for coloana, valori in erori.items():
            print(f"\nColoana '{coloana}' conține următoarele valori neconforme:")
            print(valori)


genereaza_raport_erori(check_columns_for_errors())
check_columns_for_missing_values()
check_columns_for_duplicates()