import os
import csv
import sys
from scapy.all import rdpcap, IP, TCP, UDP
from pathlib import Path

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
import pandas as pd
from send2trash import send2trash
import yaml
from dacite import from_dict
from dataclasses import asdict
import warnings
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from config import UserParams

sys.path.append("./")
warnings.simplefilter("ignore", category=UserWarning)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

User_params = from_dict(data_class=UserParams, data=params)
User_params = asdict(User_params)


def get_all_files():
    if User_params['Project'] == 'VPN':
        # lista de todos arquivos .pcap
        all_files = [name for database in root_path.iterdir() if database.is_dir()
                    for crypto_type in database.iterdir() if crypto_type.is_dir()
                    for name in crypto_type.glob("*.pcap")]
    else:
        all_files = [database for database in root_path.glob("*.pcap")]
    return all_files


def get_flow_key(pkt):
    if IP in pkt:
        proto = pkt[IP].proto
        ip_src = pkt[IP].src
        ip_dst = pkt[IP].dst

        if proto == 6 and TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif proto == 17 and UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            return None

        return (ip_src, sport, ip_dst, dport, proto)
    return None


attack_schedule = {
    datetime(2017, 7, 3).date(): [],  # Benign traffic only

    datetime(2017, 7, 4).date(): [
        {"type": "BruteForce", "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(9, 20), "end": time(10, 20)},
        {"type": "BruteForce", "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(14, 0), "end": time(15, 0)},
    ],

    datetime(2017, 7, 5).date(): [
        {"type": "DDoS",    "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(9, 47), "end": time(10, 10)},
        {"type": "DDoS", "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(10, 14), "end": time(10, 35)},
        {"type": "DDoS",         "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(10, 43), "end": time(11, 0)},
        {"type": "DDoS",    "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(11, 10), "end": time(11, 23)},
        {"type": "Exploit","src": "205.174.165.73","dst": "205.174.165.66", "start": time(15, 12), "end": time(15, 32)},
    ],

    datetime(2017, 7, 6).date(): [
        {"type": "BruteForce", "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(9, 20), "end": time(10, 0)},
        {"type": "WebAttack",        "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(10, 15), "end": time(10, 35)},
        {"type": "WebAttack",       "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(10, 40), "end": time(10, 42)},
        {"type": "Exploit", "src": "205.174.165.73", "dst": "192.168.10.8", "start": time(14, 19), "end": time(14, 35)},
        {"type": "Infiltration",  "src": "205.174.165.73", "dst": "192.168.10.25", "start": time(14, 53), "end": time(15, 0)},
        {"type": "Infiltration",   "src": "205.174.165.73", "dst": "192.168.10.8",  "start": time(15, 4),  "end": time(15, 45)},
    ],

    datetime(2017, 7, 7).date(): [
        {"type": "Botnet", "src": "205.174.165.73", "dst": "192.168.10.8",  "start": time(10, 2), "end": time(11, 2)},
        {"type": "Infiltration",    "src": "205.174.165.73", "dst": "205.174.165.68", "start": time(13, 55), "end": time(15, 29)},
        {"type": "DDoS",   "src": "205.174.165.69", "dst": "205.174.165.68", "start": time(15, 56), "end": time(16, 16)},
        {"type": "DDoS",   "src": "205.174.165.70", "dst": "205.174.165.68", "start": time(15, 56), "end": time(16, 16)},
        {"type": "DDoS",   "src": "205.174.165.71", "dst": "205.174.165.68", "start": time(15, 56), "end": time(16, 16)},
    ]
}

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



def infer_threat(pkt_time, ip_src, ip_dst):
    tz = ZoneInfo('America/Halifax')  
    dt = datetime.fromtimestamp(float(pkt_time), tz=tz)
    current_date = dt.date()
    current_time = dt.time()

    attacks_today = attack_schedule.get(current_date, [])

    def ip_match(ip_a, ip_b):
        return ip_a == ip_b or ip_b in nat_map.get(ip_a, []) or ip_a in nat_map.get(ip_b, [])

    for attack in attacks_today:
        src_match = ip_match(ip_src, attack["src"]) and ip_match(ip_dst, attack["dst"])
        dst_match = ip_match(ip_dst, attack["src"]) and ip_match(ip_src, attack["dst"])

        if (src_match or dst_match) and (attack["start"] <= current_time <= attack["end"]):
            return f"Malicious/{attack['type']}"

    return "Benign/None"



def process_pcap_to_csv(pcap_path, output_dir):
    # print(pcap_path)
    packets = rdpcap(pcap_path)

    flows = {} 

    for pkt in packets:
        key = get_flow_key(pkt)
        if not key:
            continue
        if key not in flows:
            flows[key] = []
        flows[key].append(pkt)

    os.makedirs(output_dir, exist_ok=True)

    # i = len list files from folder
    for i, (key, pkts) in enumerate(flows.items()):
        
        ip_src, sport, ip_dst, dport, proto = key
        
        start_time = pkts[0].time
        
        times = [round(pkt.time - start_time, 6) for pkt in pkts]
        sizes = [len(pkt) for pkt in pkts]

        # Inferir ameaça com base no primeiro pacote
        threat = infer_threat(start_time, ip_src, ip_dst)
        threat_dir = os.path.join(output_dir, threat)
        os.makedirs(threat_dir, exist_ok=True)

        if i == 0:
            new_flow_id = sum(1 for f in Path(threat_dir).iterdir() if f.is_file())
            i = i + new_flow_id

        flow_id = f"flow_{i}"

        row = [
            flow_id,
            ip_src,
            sport,
            ip_dst,
            dport,
            proto,
            0,  # id1 
            0,  # id2 
            0.0  # tempo inicial
        ] + times[1:] + [''] + sizes

        csv_name = os.path.join(threat_dir, f"{flow_id}.csv")
        with open(csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def get_vpn_class(pcap_name: str) -> str:
    name = pcap_name.lower()
    
    crypto_label = next((label for label, keywords in crypto.items() if any(k in name for k in keywords)), None)
    traffic_label = next((label for label, keywords in traffic.items() if any(k in name for k in keywords)), 'Browsing')

    return f"{traffic_label}_{crypto_label}" if crypto_label else traffic_label

crypto = {
    'VPN': ['vpn'],
    'nonVPN': ['nonvpn', 'nontor', 'nontor'],
    'Tor': ['tor']
}

traffic = {
    'Video': ['video', 'youtube', 'vimeo', 'netflix'],
    'VOIP': ['voip', 'voice', 'audio', 'spotify'],
    'FileTransfer': ['filetransfer', 'ftps', 'sftp', 'p2p', 'bittorrent', 'scp', 'file', 'transfer'],
    'Chat': ['chat', 'email', 'mail'],
    'Browsing': ['browsing']
}

root_path = Path('Dataset_raw/CIC-IDS-2017/')
save_database = Path('Classes_csv/')
done_csv_name = Path('./internet_traffic/files_done.csv')


# lista de arquivos .pcap ja processados
if not done_csv_name.exists():
    files_done = {
        'file_name': ['dummy']
    }
    files_done = pd.DataFrame(files_done)
    files_done.to_csv(done_csv_name.__str__())

files_done = pd.read_csv(done_csv_name.__str__())
files_done_list = files_done['file_name'].tolist()

all_files = get_all_files()

# checar tamanho e fazer split caso necessário
for file in all_files:
    file_size = file.stat().st_size / (1024 * 1024)
    if file_size > 500:
        command = f'editcap -c 100000 {file} {file}'
        os.system(command)
        send2trash(file)

# atualiza lista de todos arquivos .pcap
all_files = get_all_files()

# lista de arquivos .pcap nao processados
not_done_list = [str(item) for item in all_files if str(item) not in files_done_list]


# barra de progresso
progress = Progress(
    TextColumn("[bold blue]Total Progress:"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
)
task = progress.add_task("Processing PCAPs", total=len(all_files))

progress.update(task, completed=(len(all_files) - len(not_done_list)))


# processando cada arquivo ainda nao processado
with Live(progress, refresh_per_second=2):

    for file in not_done_list:
        if User_params['Project'] == 'VPN':
            crypto_type, pcap_file = (str(file).split('\\')[2], str(file).split('\\')[3])
            pcap_name = f"{crypto_type}\\{pcap_file}".lower()

            pcap_class = get_vpn_class(pcap_name)

            process_pcap_to_csv(str(file), str(save_database.joinpath(pcap_class.replace('_', '/'))))

            files_done.loc[len(files_done)] = [len(files_done), str(file)]
            files_done.to_csv(done_csv_name.__str__(), index=False)
            files_done_list = files_done['file_name'].tolist()
            
            progress.update(task, advance=1)
        else:
            # category, threat = (str(file).split('\\')[2], str(file).split('\\')[3])
            # pcap_class = f"{category.title()}/{threat.title()}"

            # pcap_class = get_vpn_class(pcap_name)
            
            process_pcap_to_csv(str(file), str(save_database))

            files_done.loc[len(files_done)] = [len(files_done), str(file)]
            files_done.to_csv(done_csv_name.__str__(), index=False)
            files_done_list = files_done['file_name'].tolist()
            
            progress.update(task, advance=1)