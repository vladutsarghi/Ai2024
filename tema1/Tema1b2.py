
import pandas as pd

df = pd.read_excel("C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx")

print("Tipurile de date ale coloanelor:")
print(df.dtypes)

df['Race'] = pd.to_numeric(df['Race'], errors='coerce')  # Asigură-te că 'Breed' este numeric

print("Tipul coloanei 'Race':", df['Race'].dtype)
print("Valorile unice din coloana 'Race':")
print(df['Race'].unique())

df_numeric = df.select_dtypes(include='number')

correlation = df_numeric.corr()

breed_correlation = correlation['Race']

breed_correlation_df = breed_correlation.drop('Race')

breed_correlation_sorted = breed_correlation_df.sort_values(ascending=False)

breed_correlation_sorted.to_excel('corelatie_breed_atribute_sortate.xlsx', sheet_name='Corelație', index=True)

print("Corelațiile dintre 'Race' și celelalte atribute (ordonate descrescător):")
print(breed_correlation_sorted)

print("Corelațiile ordonate au fost salvate în 'corelatie_breed_atribute_sortate.xlsx'.")