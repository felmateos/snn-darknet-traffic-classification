# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:59:37 2020

@author: Florian Delpech

This program enables to process dataset containing traffic flows.
Input: csv files with traffic data (i.e sessions) distributed in labelled folders (ex: Browsing/Tor or Video/VPN)
Outputs:
    * The full dataset with all processed samples (in csv files) in the folder Dataset
    * The images created for each sample in the folder Pictures

Here, a session stands for an exchange of data between a source and a destination (designated by IP adresses and ports)
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

nat_map = {
    # Atacante principal (Kali)
    "205.174.165.73": [
        "192.168.10.8",     # Windows Vista
        "192.168.10.25",    # MAC
        "192.168.10.50",    # WebServer Ubuntu
        "192.168.10.51"     # Ubuntu12 (Heartbleed)
    ],

    # Vítimas internas mapeadas para o IP público 205.174.165.73 (respostas)
    "192.168.10.8": ["205.174.165.73"],
    "192.168.10.25": ["205.174.165.73"],
    "192.168.10.50": ["205.174.165.73"],
    "192.168.10.51": ["205.174.165.73"],

    # WebServer Ubuntu com IP público da firewall (NAT)
    "205.174.165.80": [
        "192.168.10.50",    # WebServer Ubuntu
        "192.168.10.51"     # Ubuntu12
    ],
    "192.168.10.50": ["205.174.165.80"],
    "192.168.10.51": ["205.174.165.80"],

    # IPs públicos das máquinas da botnet (LOIT)
    "205.174.165.69": ["192.168.10.50"],
    "205.174.165.70": ["192.168.10.50"],
    "205.174.165.71": ["192.168.10.50"],

    # Relacionamentos anteriores já registrados
    "205.174.165.66": ["192.168.10.5"],
    "192.168.10.5": ["205.174.165.66"],
}

def ip_equiv(ip1, ip2, nat_map):
    # Retorna True se ip1 e ip2 são equivalentes via NAT
    if ip1 == ip2:
        return True
    if ip1 in nat_map and nat_map[ip1] == ip2:
        return True
    if ip2 in nat_map and nat_map[ip2] == ip1:
        return True
    return False

# A function that enables to get the key from the value that corresponds to it
def key(dictio,val):
    for c,v in dictio.items():
        if v == val:
            return(c)

# A function that returns the index of the separation between the list of times of arrival and the list of packet sizes (marked by a space)
def ind_split(row, elt):
    for i in range(len(row)):
        if row[i] == elt:
            return i
    return -1

# A function that creates a dictionnary of all IP adresses that can be found in the different sessions
def search_ip(dico, IP):
    for ip in IP:
        test = False
        ind = 0
        while not test and ind < len(dico):
            if key(dico,ind) == ip:
                test = True
            ind += 1
        if not test:
            dico.update({ip:len(dico)})
            dico_writer.writerow([ip,len(dico)])
    return 

# A function that cheks that every element in a given list l can be found only once (and print it if not)
def check_unicity(l):
    the_list = np.unique(l, return_counts=True)
    for i in range(len(the_list[1])):
        if the_list[1][i] != 1:
            # print(the_list[0][i])
            return False
    return True

# A function that gives the name of the application, the IP adresses and the port numbers for one given session
def data(session):
    App = session.split("_")[2]
    IP = session.split("_")[3].split("(")[1].replace(")","").split("-")
    IP = (int(IP[0]), int(IP[1]))
    Port = session.split("_")[4].split("(")[1].replace(")","").split("-")
    Port = (int(Port[0]), int(Port[1]))
    return App, IP, Port

# A function that, given a session, returns the paired up session (IP adresses and port numbers inverted)
def pairing(list_session, session, nat_map):
    App, IP, Port = data(session)
    for session2 in list_session:
        app, ip, port = data(session2)
        # if App != app:
        #     continue
        if (
            ip_equiv(str(IP[0]), str(ip[1]), nat_map) and
            ip_equiv(str(IP[1]), str(ip[0]), nat_map) and
            Port[0] == port[1] and
            Port[1] == port[0]
        ):
            list_session.remove(session)
            list_session.remove(session2)
            return (session, session2)
    list_session.remove(session)
    return "Single"

    
IP_adresses = {}
names = {} # Keys = labels ; Values = List of session names
Pairs = {} # Keys = labels ; Values = List of paired sessions
Values = {} # Keys = session names : Values = Data (dictionary containing packet' size list and time of arrival list)

dico_ip = open("ip_adresses.csv", "w", newline = '')
dico_writer = csv.writer(dico_ip)

for root,dirs,files in os.walk("./"):
    
    if "Classes_csv" not in root or len(files) == 0 :
        continue
    
    label = "_".join([root.split('\\')[1], root.split('\\')[2]])
    print(label, end ="\n")
    count = 0
    names.update({label:[]})
    Pairs.update({label:[]})
    
    for file in files:
        
        if ".csv" not in file: ##############################################
            continue 
        
        # print(file)
        
        path = "\\".join([root,file])
        input_file = open(path, "r")
        reader = csv.reader(input_file)
        
        for row in reader:
            count += 1
            print("Processing file", row[0], count)
            assert(row[8] == '0.0') #first time_of_arrival
            Raw = row[:8]
            search_ip(IP_adresses, [row[1], row[3]]) # ip adresses from src and dst
            ind = ind_split(row, '')
            assert(ind > 0)
            Time_of_arrival = list(map(float, row[8:ind]))
            Packet_size = list(map(int, row[ind+1:])) #packet size is the first col?
            assert(len(Time_of_arrival) == len(Packet_size))
            name = label + "_" + Raw[0].replace("_","") + "_(" + str(IP_adresses[row[1]]) + "-" + str(IP_adresses[row[3]]) + ")_(" + str(row[2]) + "-" + str(row[4]) + ")"
            names[label].append(name)
            Values.update({name:{'Time_of_arrival':Time_of_arrival, 'Packet_size':Packet_size}})
            # print({name:{'Time_of_arrival':Time_of_arrival, 'Packet_size':Packet_size}}) # added after
        
        input_file.close()
    
    assert(check_unicity(names[label]))
    
        
        
    while len(names[label]) !=0 :
        session = names[label][0]
        tuple_pairs = pairing(names[label], session, nat_map)
        if "Single" in tuple_pairs:
            Pairs[label].append((session, "None"))
        else:
            Pairs[label].append(tuple_pairs)
    
    print()
    
TPS = 60 # Time Per Session
DELTA_T = 15 # Interval between sessions
MIN_TPS = 0.00001 # Minimum time per session
MIN_LENGHT = 1 # Minimum number of packets for one session

# A function that builds the histograms according to the constraints (above values)
def build_histogram(list_ind, times, sizes):
    if len(times) == 0:
        return "NULL", "NULL"
    if type(times[0]) == int:
        return times, sizes
    if len(list_ind) < MIN_LENGHT:
        print(f"Descartado: poucos pacotes ({len(list_ind)} < {MIN_LENGHT})")
        return "NULL", "NULL"
    first_ind, last_ind = list_ind[0], list_ind[-1]
    Time, Size = times[first_ind:last_ind+1], sizes[first_ind:last_ind+1]
    if Time[-1]-Time[0] < MIN_TPS:
        return "NULL", "NULL"
    return Time, Size
        
# A function that enable to separate simultaneously paired sessions into several one according to TPS
def splitting(pairs):
    times1, sizes1 = Values[pairs[0]]["Time_of_arrival"], Values[pairs[0]]["Packet_size"]
    if "None" in pairs[1]:
        times2, sizes2 = [0], [0] # if there is not paired session, an empty histogram is built 
    else:
        times2, sizes2 = Values[pairs[1]]["Time_of_arrival"], Values[pairs[1]]["Packet_size"]
        
    time_max = max(times1[-1], times2[-1])
    rng = int(time_max/DELTA_T - TPS/DELTA_T) + 1
    if rng <= 0:
        rng = 1
    for t in range(rng):
        ind_times1 = [i for i in range(len(times1)) if ((times1[i] >= t*DELTA_T) and (times1[i] <= (t*DELTA_T+TPS)))]
        ind_times2 = [i for i in range(len(times2)) if ((times2[i] >= t*DELTA_T) and (times2[i] <= (t*DELTA_T+TPS)))]
        Time1, Size1 = build_histogram(ind_times1, times1, sizes1)
        Time2, Size2 = build_histogram(ind_times2, times2, sizes2)
        if Time1 == "NULL" or Time2 == "NULL":
            continue
        name = "_".join([pairs[0], str(t)])
        split_dict.update({name:{"1_Time_of_arrival":Time1, "1_Packet_size": Size1, "2_Time_of_arrival":Time2, "2_Packet_size":Size2}})
        
        name_writer.writerow([name])
        
        data_file = open("Data/" + name + ".csv", "w", newline='')
        data_writer = csv.writer(data_file)
        data_writer.writerows([Time1,Size1,Time2,Size2])
        data_file.close()
        save_image(name+"_(1)", Time1, Size1)
        save_image(name+"_(2)", Time2, Size2)
        
    return 

# A function that saves the built images for each session
def save_image(name,Time,Size):
    plt.plot(Time, Size, "k.")
    plt.xlabel("Time of arrival")
    plt.ylabel("Packet size")
    plt.title(name + "\n " + str(len(Size)) + " paquets ; " + str(round(Time[-1]-Time[0],1)) + "s")
    plt.savefig("Pictures/" + name.split("_")[0] + "/" + name.split("_")[1] + "/" + name + ".png")
    plt.clf()
    
name_list = open("Dataset.csv", "w", newline='')
name_writer = csv.writer(name_list)
split_dict = {}
count = 0
for label,list_names in Pairs.items():
    print(label)
    for i in range(len(list_names)):
        count += 1
        print("Processing data", count)
        splitting(list_names[i])
    print()
name_list.close()
dico_ip.close()

    




