#Transformarea valorilor non-numerice ale atributelor în valori numerice.
#Afișarea grafică a distribuției valorilor pentru fiecare atribut (histograme, boxplot).

import pandas as pd

file_path = "C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx"
xls = pd.ExcelFile(file_path)

data_df = pd.read_excel(xls, sheet_name='Data')
code_df = pd.read_excel(xls, sheet_name='Code')

age_dict = {
    "Moinsde1": 1,
    "1a2": 2,
    "2a10": 3,
    "Plusde10": 4
}

race_dict = {
    "BEN": 1,
    "SBI": 2,
    "BRI": 3,
    "CHA": 4,
    "EUR": 5,
    "MCO": 6,
    "PER": 7,
    "RAG": 8,
    "SPH": 9,
    "ORI": 10,
    "TUV": 11,
    "Autre": 12,
    "NSP": 13
}

logement_dict = {
    "ASB" : 1,
    "AAB" : 2,
    "ML" : 3,
    "MI" : 4
}

zone_dict = {
    "U" : 1,
    "PU" : 2,
    "R" : 3
}

numbre_dict = {
    "Plusde5" : 6
}

sexe_dict = {
    "M" : 0,
    "F" : 1
}


xls = pd.read_excel("C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx", sheet_name=None)
df_sheet1 = xls['Data']

df_sheet1['Age'] = data_df['Age'].replace(age_dict)
df_sheet1['Race'] = data_df['Race'].replace(race_dict)
df_sheet1['Logement'] = data_df['Logement'].replace(logement_dict)
df_sheet1['Zone'] = data_df['Zone'].replace(zone_dict)
df_sheet1['Nombre'] = data_df['Nombre'].replace(numbre_dict)
df_sheet1['Sexe'] = data_df['Sexe'].replace(sexe_dict)

xls['Data'] = df_sheet1

with pd.ExcelWriter("C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx", engine='openpyxl') as writer:
    for sheet_name, df_sheet in xls.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)





