import os
import csv
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def make_df_with_all_results(pfad):
    header = []
    daten = []

    pfad = pfad
    dateien = os.listdir(pfad)

    for datei in dateien:
        with open(f'{pfad}/{datei}', 'r') as readed:
            reader = csv.reader(readed, delimiter=',')
            header = next(reader)
            daten.extend([row for row in reader])

    with open('gesamt.csv', 'w') as ergebnis_datei:
        writer = csv.writer(ergebnis_datei, delimiter=',')
        writer.writerow(header)
        writer.writerows(daten)
    
    dataset = pd.read_csv('gesamt.csv', index_col= 0)

    return dataset

def plot_corr_all_stadium(data_stadiums, data_names,method):
    
    all_corr = pd.DataFrame()
    count = 521
    counter = 0
    for i in data_stadiums:
        corr = i[i.columns[1:-1]].apply(lambda x: x.corr(i['dia'],method=method))
        neww = corr.sort_values(ascending=False)
        df = pd.DataFrame(data=neww, index=None, columns=['corr'], dtype=None, copy=None)
        #fig, ax =plt.subplots(1,2,3,4,5)
        all_corr.insert(counter, column=data_names[counter],value= df['corr'])
        
        sb.set(rc = {'figure.figsize':(16,70)})
        plt.subplot(count)
        plt.title(method+' ' +data_names[counter]+' '+str(count))
        sb.heatmap(df, 
            xticklabels=df.columns,
            yticklabels=df.index,
            cmap='cubehelix',
            #cmap='RdBu_r',
            annot=True,
            linewidth=1)
        count = count + 1
        counter = counter +1
    
    return all_corr

        
def style_negative(v, props=''):
    return props if v < 0 else None

def make_different_df(stadium_name, dataset):
    
    stadium = dataset.loc[dataset['stadium'] == stadium_name]
    stadium.reset_index(inplace=True, drop=True)
    
    stadium_dia_one = stadium.loc[stadium['dia'] == 1]
    stadium_dia_two = stadium.loc[stadium['dia'] == 2]
    stadium_dia_three = stadium.loc[stadium['dia'] == 3]
    
    return stadium, stadium_dia_one, stadium_dia_two, stadium_dia_three 

def make_all_largest(all_corr):

    large_wach = all_corr.nlargest(5, 'Wach',keep='all')
    large_rem = all_corr.nlargest(5, 'Rem',keep='all')
    large_n1 = all_corr.nlargest(5, 'N1',keep='all')
    large_n2 = all_corr.nlargest(5, 'N2',keep='all')
    large_n3 = all_corr.nlargest(5, 'N3',keep='all')

    all_large = pd.DataFrame(data=[large_wach['Wach'],large_rem['Rem'],large_n1['N1'],large_n2['N2'],large_n3['N3']])
                             #columns=['Wach','Rem', 'N1', 'N2','N3'])
    all_large = all_large.round(3)
    all_large = all_large.transpose()
    
    return all_large

def make_all_smallest(all_corr):

    small_wach = all_corr.nsmallest(5, 'Wach',keep='all')
    small_rem = all_corr.nsmallest(5, 'Rem',keep='all')
    small_n1 = all_corr.nsmallest(5, 'N1',keep='all')
    small_n2 = all_corr.nsmallest(5, 'N2',keep='all')
    small_n3 = all_corr.nsmallest(5, 'N3',keep='all')

    all_small = pd.DataFrame(data=[small_wach['Wach'],small_rem['Rem'],small_n1['N1'],small_n2['N2'],small_n3['N3']])
                             #columns=['Wach','Rem', 'N1', 'N2','N3'])
    all_small = all_small.round(3)
    all_small = all_small.transpose()
    
    return all_small
