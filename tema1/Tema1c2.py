import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/sargh/Desktop/AIgit/tema1/data.xlsx")

print(df.columns)

output_folder = "C:/Users/sargh/Desktop/AIgit/tema1/"


for column in df.select_dtypes(include=['int64', 'float64']).columns:
    if column != 'Race':
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Race', y=column, data=df)
        plt.title(f'Comparative Boxplot pentru {column} între toate rasele')
        plt.xlabel('Race')
        plt.ylabel(column)
        plt.grid(True)
        plt.savefig(output_folder + f"boxplot_comparativ_{column}.png")

        plt.show()
