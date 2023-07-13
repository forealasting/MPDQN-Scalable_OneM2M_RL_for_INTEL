
# r = 7
# path = "request" + str(r) + ".txt"
path1 = "request13.txt"
#
# f = open(path, "r")
f1 = open(path1, 'a')

# request = []
# tmp_data = 0
# for line in f:
#     data = float(line)
#
#     data = data/5
#     data = int(data)
#
#     # if data != int(tmp_data):
#     #     # print(data, tmp_data)
#     request.append(data)
#
#     tmp_data = data


# f.close()
# print(request)
# print(len(request))
# req_m = []
#
# request = []
# idx = 10
# done = 1
# for i in range(6):
#     for j in range(50):
#         request.append(idx)
#     idx += 10
# print(request)
#
#
# for i in request:
#
#     req_m.append(i)
#     data = str(i) + '\n'
#     f1.write(data)
# f1.close()

# for i in request:
#     # for j in range(6):
#     req_m.append(i)
#     data = str(i) + '\n'
#     f1.write(data)
# f1.close()



def generate_data_rate_pattern(total_time):
    data_rate_pattern = []
    timestamp = 0
    data_rate = 20
    increasing = True

    while timestamp < total_time:
        if data_rate == 100:
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
pattern = generate_data_rate_pattern(3600)
print(len(pattern))

for i in pattern:
    data = str(i) + '\n'
    f1.write(data)
f1.close()