import csv
import tools.datatools as dto

reader = csv.reader(open("../data/201802171302.txt", 'r'),delimiter='|')

index = 0
print(reader)
for row in reader:
    #此时输出的是一行行的列表
    print(row)
    print(type(row))
    for i in range(0, len(row)):
        print(row[i], type(row[i]), dto.getTypeName(dto.getTypeOf(row[i])))

    index += 1
    if index > 10:
        break
