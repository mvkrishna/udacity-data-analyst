import csv
with open('titanic_data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if float(row['Age']) <= float(20):
            print '0-20'
        elif float(row['Age']) > float(20) and float(row['Age']) <= float(40):
            print '20-40'
        elif float(row['Age']) > float(40) and float(row['Age']) <= float(60):
            print '40-60'
        elif float(row['Age']) > float(60) and float(row['Age']) <= float(80):
            print '60-80'
        else:
            print '80+'
