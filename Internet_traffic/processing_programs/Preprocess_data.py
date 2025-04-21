import os
import csv
from scapy.all import rdpcap, IP, TCP, UDP
from pathlib import Path


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
            return None  # ignore if not TCP nor UDP

        return (ip_src, sport, ip_dst, dport, proto)
    return None


def process_pcap_to_csv(pcap_path, output_dir):
    packets = rdpcap(pcap_path)

    flows = {}  # groups by 5-tuple

    for pkt in packets:
        key = get_flow_key(pkt)
        if not key:
            continue
        if key not in flows:
            flows[key] = []
        flows[key].append(pkt)

    os.makedirs(output_dir, exist_ok=True)
    for i, (key, pkts) in enumerate(flows.items()):
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
            0,  # id1 (placeholder)
            0,  # id2 (placeholder)
            0.0  # initial relative time (always 0)
        ] + times[1:] + [''] + sizes

        csv_name = os.path.join(output_dir, f"{flow_id}.csv")
        with open(csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[✓] Exported {csv_name} with {len(pkts)} packets.")


pcap_path = Path(r'Dataset_raw\ISCX-VPN-nonVPN-2016\VPN-PCAPS-01')
save_dir_path = Path("Classes_csv/")


pcap_categories = {
    'vpn_aim_chat1a.pcap': 'Chat_VPN',
    'vpn_aim_chat1b.pcap': 'Chat_VPN',
    'vpn_bittorrent.pcap': 'FileTransfer_VPN',
    'vpn_email2a.pcap': 'Chat_VPN',
    'vpn_email2b.pcap': 'Chat_VPN',
    'vpn_facebook_audio2.pcap': 'VOIP_VPN',
    'vpn_facebook_chat1a.pcap': 'Chat_VPN',
    'vpn_facebook_chat1b.pcap': 'Chat_VPN',
    'vpn_ftps_A.pcap': 'FileTransfer_VPN',
    'vpn_ftps_B.pcap': 'FileTransfer_VPN',
    'vpn_hangouts_audio1.pcap': 'VOIP_VPN',
    'vpn_hangouts_audio2.pcap': 'VOIP_VPN',
    'vpn_hangouts_chat1a.pcap': 'Chat_VPN',
    'vpn_hangouts_chat1b.pcap': 'Chat_VPN'
}


for pcap_file in pcap_path.glob("*.pcap"):
    filename = pcap_file.__str__().split('VPN-PCAPS-01\\')[1]
    process_pcap_to_csv(pcap_file.__str__(), save_dir_path.joinpath(pcap_categories.get(filename).replace('_', '/')))
