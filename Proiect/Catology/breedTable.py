import openpyxl
from collections import defaultdict
import pandas as pd

def count_values_per_attribute():
    file_path = 'static/final4.xlsx'
    data = pd.read_excel(file_path)

    breeds_id = data.iloc[:, 1].unique()

    print(breeds_id)
    not_good_i = [0, 1, 3, 8, 10, 14, 15, 17, 18, 19, 20, 22]


    for breed in range(1, 13):
        filtered_data = data[data.iloc[:, 1] == breed]  # Selectăm rândurile unde coloana 2 este 1
        for i in range(24):
            if i not in not_good_i:
                count_of_zeros = (filtered_data.iloc[:, i] == 0).sum()
                count_of_ones = (filtered_data.iloc[:, i] == 1).sum()
                count_of_twos = (filtered_data.iloc[:, i] == 2).sum()
                count_of_threes = (filtered_data.iloc[:, i] == 3).sum()
                count_of_fours = (filtered_data.iloc[:, i] == 4).sum()
                # Afișăm rezultatele

                avg = ((count_of_zeros + count_of_ones + 2 * count_of_twos + 3 * count_of_threes + 4 * count_of_fours) /
                       (count_of_ones + count_of_twos + count_of_threes + count_of_fours))


# Apelarea funcției

count_values_per_attribute()