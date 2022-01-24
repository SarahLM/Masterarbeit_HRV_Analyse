# HRV Analyse

Dieses Projekt ist im Rahmen einer Masterarbeit zum Thema Herzratenvariabilität bei Narkolepsie entstanden. Ziel ist, eine Analyse der Herzratenvariabilität aus Polysomnografiedaten zu ermöglichen und sie anschließend als Features für verschiedene Machine Learning Algorithmen zu nutzen.


Dieses Projekt beinhaltet insgesamt fünf Notebooks:

Für das Feature Engeneering(abhängig vom System der Ursprungsdaten):
- HRV_analysis-alice5.ipynb
- HRV_analysis-nihon-kohden.ipynb
- HRV_analysis-somnomedics.ipynb

Für das Testing:
- testing.ipynb

Für das ML:
- statistik_ml.ipynb

Zur Auführung des Projekts, muss Jupyter Notebook installiert sein, eine Anleitung findet sich [hier](https://jupyter.org/install).


Darüber hinaus muss Python 3 auf dem System oder in der Virtuellen Umgebung installiert sein, je nach gewählter Ausführungsart von Jupyter Notebook. 

## Beispiel

Beispieldaten finden sich für die Hypnogramme im Repository: 

- Somnomedics = "Schlafprofil.txt"
- Nihon Kohden = "Nihon_hypno_anon.html"
- Alice5 = "STADIUManonym_alice.csv"

Eine Beispiel EDF kann unter [zenodo](https://zenodo.org/record/5895263) heruntergeladen werden, passend für den Ablauf von Somnomedics. 
Für die beiden anderen kann leider aus Datenschutzgründen keine Beispieldatei bereitgestellt werden. 


### Beispiel Ablauf: 

Im folgenden soll ein möglicher Abalauf der Analyse und Auswertung der Schlafstadien und des Hypnogramms anhand der `somnomedics` Dateien gezeigt werden.

1. Die heruntergeldenen Beispieldaten werden in in den dazugehörigen Ordner `Probanden` abgelegt.
1. danach wird die Datei `hrv_analysis-somnomedics.ipynb` aufgerufen
1. Um das Script durchlaufen zu lassen wird in Jupyter Notebook auf das __play__ Symbol gedrückt.



## Testing 

Um die mit Doctest erstellten Tests auszuführen. ist das Vorgehen ähnlich wie oben beschrieben:

1. Für die Tests werden die Beispieldaten benötigt. 
1. Es wird die Datei `testting.ipynb` geöffnent, die den Aufruf der Doctest beeinhaltet. 
1. Um die Tests zu starten, wird in Jupyter Notebook auf das __play__ Symbol gedrückt.

### Verbose
```python
doctest.testmod(prepedf.FILE_NAME, verbose=False)
````

Der Auruf der Tests hat noch ein zusätzliches `verbose` Attribute, was auf `False` gestellt ist. Wird der Parameter geändert, so wird bei jeden Testfall, welcher durchläuft wird, der Test selbst, das zu erwartene Ergebnis und das Ergebnis selbst angezeigt.   



## Mögliche Fehler: 

Beim Lesen von einer CSV Datei mit der Panadas Bibliothek kann es zu einem möglichen Fehler kommen, dass die übergebenen Paramater 
`delimiter` und `sep` nicht beide vorhanden sein dürfen. Der Fehler ist ggf. auf unterschiedliche Python und Paket
Versionen zurückzuführen.

Dazu in die Funktion `def fileopener(filename):` gehen, die sich in der `preperation_hypno.py` befindet und folgende Zeile:
```python 
return pd.read_csv(filename, skiprows=5, delimiter=";", sep=" ", names=["Time", "Stadium"], header=0)
```

durch diese ersetzen:

```python 
return pd.read_csv(filename, skiprows=5, delimiter=";", names=["Time", "Stadium"], header=0)
```

# Lizent

Das Projekt steht unter der [MIT](./LICENSE.md) Lizenz.
