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
        all_files = [name for database in root_path.iterdir() if database.is_dir()
                    for category in database.iterdir() if category.is_dir()
                    for threat in category.iterdir() if threat.is_dir()
                    for name in threat.glob("*.pcap")]
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


def process_pcap_to_csv(pcap_path, output_dir):
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

    new_flow_id = sum(1 for f in Path(output_dir).iterdir() if f.is_file())

    # i = len list files from folder
    for i, (key, pkts) in enumerate(flows.items()):
        if i == 0:
            i = i + new_flow_id
        ip_src, sport, ip_dst, dport, proto = key
        flow_id = f"flow_{i}"
        start_time = pkts[0].time
        times = [round(pkt.time - start_time, 6) for pkt in pkts]
        sizes = [len(pkt) for pkt in pkts]

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

        csv_name = os.path.join(output_dir, f"{flow_id}.csv")
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

root_path = Path('Dataset_raw/')
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
            category, threat = (str(file).split('\\')[2], str(file).split('\\')[3])
            pcap_class = f"{category.title()}/{threat.title()}"

            # pcap_class = get_vpn_class(pcap_name)

            process_pcap_to_csv(str(file), str(save_database.joinpath(pcap_class)))

            files_done.loc[len(files_done)] = [len(files_done), str(file)]
            files_done.to_csv(done_csv_name.__str__(), index=False)
            files_done_list = files_done['file_name'].tolist()
            
            progress.update(task, advance=1)