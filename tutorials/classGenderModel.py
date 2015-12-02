import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../csv/train.csv'))
header = next(csv_file_object)

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

# Cap fares at 39
fare_ceiling = 40
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = int(fare_ceiling / fare_bracket_size)

number_of_classes = len(np.unique(data[0::,2]))

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))
for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        women_only_stats = data[                        \
                #is a woman
                (data[0::,4] == "female")               \
                #in i-th class
                &(data[0::,2].astype(np.float) == i+1)  \
                #and the right fare bin
                &(data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                &(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) \
                , 1]

        men_only_stats = data[                          \
                #is a man
                (data[0::,4] != "female")               \
                #with i-th class
                &(data[0::,2].astype(np.float) == i+1)  \
                #and the right fare bin
                &(data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                &(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) \
                , 1]

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

#fix nans
survival_table[survival_table != survival_table] = 0

#if prop under .5, they ded
survival_table[survival_table < 0.5] = 0
survival_table[survival_table > 0.5] = 1

test_file = open('../csv/test.csv')
test_file_object = csv.reader(test_file)
header = next(test_file_object)

prediction_file = open("../csv/genderclassmodel.csv", "w")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8] = float(row[9])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets - 1
            break
        if row[8] >= j*fare_bracket_size\
                and row[8] <            \
                (j+1) * fare_bracket_size:
            bin_fare = j
            break
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], "%d" % \
                int(survival_table[0, float(row[1])-1, bin_fare])])
    else:
        prediction_file_object.writerow([row[0], "%d" % \
                int(survival_table[1, float(row[1])-1, bin_fare])])

test_file.close()
prediction_file.close()
