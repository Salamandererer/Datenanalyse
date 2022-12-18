import numpy as np
import pandas as pd
from linkmethods import get_forward_links,get_back_links
from datetime import datetime

def to_links(data: dict, prop: str):
    data = data["query"]
    data = data["pages"]
    data = data[list(data.keys())[0]]
    return data[f"{prop}"]

##############################################
if __name__ == '__main__':
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    bundestagswahl2 = get_back_links("Bundestagswahl 2021")
    print(bundestagswahl2)
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    meisen = get_back_links("Meisen")

    print(meisen)
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

    df = pd.DataFrame(meisen)
    df.columns = ['referring links']
    df['link number'] = np.arange(len(df))
    df['links to'] = 'Meisen'
    df = df.set_index('links to')
    df.to_csv('Meisennnnnnnnnnnnnnnnnnnnnnnn.csv', index=True)
    print("MEISEN HIER BACKLINS", get_back_links("Meisen"))

    print(df)
