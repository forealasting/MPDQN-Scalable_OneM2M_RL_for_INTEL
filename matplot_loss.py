import re
import matplotlib.pyplot as plt
tmp_dir = "mpdqn_result/result9/"


path1 = tmp_dir +"app_mn1_actor_loss.txt"
path2 = tmp_dir +"app_mn1_critic_loss.txt"
path3 = tmp_dir +"app_mn2_actor_loss.txt"
path4 = tmp_dir +"app_mn2_critic_loss.txt"
path_list = [path1, path2, path3, path4]

loss_type = ["actor", "critic"]
service = ["First_level_MNCSE", "Second_level_MNCSE"]
# with open('loss/app_mn1_actor_loss.txt', 'r') as file:
#
#     data = [float(re.search(r'-?\d+\.\d+', line).group()) for line in file]

# generate index
def fig_add_loss(x, y, service_name, loss_type):
    # plot
    plt.plot(x, data)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.title(service_name + " " + loss_type)
    plt.savefig(tmp_dir + " " + service_name + " " + loss_type +"_loss.png", dpi=300)
    #plt.title('critic_loss')
    plt.show()

tmp_count = 0
for p in path_list:
    with open(p, 'r') as file:
        data = [float(re.search(r'-?\d+\.\d+', line).group()) for line in file]
        x = list(range(len(data)))

    fig_add_loss(x, data, service[int(tmp_count/2)], loss_type[int(tmp_count%2)])
    tmp_count += 1
