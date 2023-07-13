
# r = 7
# path = "request" + str(r) + ".txt"
path1 = "request14.txt"
#
# f = open(path, "r")
f1 = open(path1, 'a')





def generate_data_rate_pattern(total_time):
    data_rate_pattern = []
    timestamp = 0
    data_rate = 20
    increasing = True

    while timestamp < total_time:
        if data_rate == 80:
            increasing = False
        elif data_rate == 20:
            increasing = True

        data_rate_pattern.extend([data_rate] * 240)
        timestamp += 240
        if increasing:
            data_rate += 20
        else:
            data_rate -= 20

    return data_rate_pattern[:total_time]

#
pattern = generate_data_rate_pattern(3660)
print(len(pattern))

for i in pattern:
    data = str(i) + '\n'
    f1.write(data)
f1.close()