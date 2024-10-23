#(0.5p) Afișarea numărul de instanțe pentru fiecare clasă (rasă de pisici).
# Afișarea listei de valori distincte pentru fiecare atribut și să a numărului total de valori și frecvența pentru fiecare valoare,
# la nivelul întregului fișier și la nivelul fiecărei clase. Identificarea corelației între atribute și clase.
import pandas as pd

file_path = "C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx"
xls = pd.ExcelFile(file_path)

data_df = pd.read_excel(xls, sheet_name='Data')
code_df = pd.read_excel(xls, sheet_name='Code')


def print_all_cat_breeds():
    breeds=data_df['Race'].value_counts()
    print(f"numărul de instanțe pentru fiecare clasă: {breeds}")
print_all_cat_breeds()

print("\nValori distincte pentru fiecare atribut si frecventele lor:")
for column in data_df.columns[2:28]:
    distinct_values = data_df[column].value_counts()
    print(f"{column}: {distinct_values}")

print("\nValori distincte pentru fiecare atribut la nivelul fiecarei rase de pisici:")
for breed in data_df['Race'].unique():
    filtered_df = data_df[data_df['Race'] == breed]
    for column in data_df.columns[2:28]:
        if column != 'Race':
            distinct_values_class = filtered_df[column].value_counts()
            print(f"\n{column} pentru rasa {breed}: {distinct_values_class}")
