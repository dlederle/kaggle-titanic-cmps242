import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../csv/train.csv'))
header = next(csv_file_object)

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived = \
                        np.sum(women_onboard) / np.size(women_onboard)

proportion_men_survived = \
                        np.sum(men_onboard) / np.size(men_onboard)


test_file = open('../csv/test.csv')
test_file_object = csv.reader(test_file)
header = next(test_file_object)


prediction_file = open("../csv/genderbasedmodel.csv", "w")
prediction_file_object = csv.writer(prediction_file)


prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0],'1'])
    else:
        prediction_file_object.writerow([row[0],'0'])

test_file.close()
prediction_file.close()
