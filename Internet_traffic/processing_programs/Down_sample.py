import os
import shutil
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

def extract_features(csv_path):
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None  # arquivo inválido

            time_ida = np.array(list(map(float, lines[0].strip().split(","))))
            size_ida = np.array(list(map(float, lines[1].strip().split(","))))
            
            if len(lines) >= 4 and not (lines[2].strip() == "0" and lines[3].strip() == "0"):
                time_volta = np.array(list(map(float, lines[2].strip().split(","))))
                size_volta = np.array(list(map(float, lines[3].strip().split(","))))
            else:
                time_volta = np.array([])
                size_volta = np.array([])

        def stats(time, size):
            if len(time) < 2:
                duration = 0
                pkt_rate = 0
            else:
                duration = time[-1] - time[0]
                pkt_rate = len(time) / duration if duration > 0 else 0

            return [
                len(time),
                duration,
                np.mean(size) if len(size) > 0 else 0,
                np.std(size) if len(size) > 0 else 0,
                pkt_rate
            ]

        return stats(time_ida, size_ida) + stats(time_volta, size_volta)

    except Exception as e:
        print(f"Erro ao processar {csv_path}: {e}")
        return None


def filter_similar_csvs(folder_path, output_folder, similarity_threshold=0.95):
    from tqdm import tqdm
    os.makedirs(output_folder, exist_ok=True)

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith(".csv")) & (str(f).__contains__('Benign_None') or str(f).__contains__('Malicious_Infiltration'))]

    pass_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith(".csv")) & (str(f).__contains__('Malicious_Botnet') or str(f).__contains__('Malicious_Exploit'))]
    
    features = []
    valid_files = []
    for path in tqdm(file_paths, desc="Extraindo características"):
        f = extract_features(path)
        if f:
            features.append(f)
            valid_files.append(path)

    if len(features) < 2:
        print("Poucos arquivos válidos para comparar.")
        return

    features = np.array(features)
    sim_matrix = cosine_similarity(features)
    sim_matrix = np.clip(sim_matrix, 0, 1)
    distance_matrix = 1 - sim_matrix
    clustering = DBSCAN(eps=1-similarity_threshold, min_samples=1, metric='precomputed')
    distance_matrix = 1 - sim_matrix
    labels = clustering.fit_predict(distance_matrix)

    to_keep = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        to_keep.append(valid_files[indices[0]])  # salva só o primeiro do grupo
    
    for label in pass_paths:
        print(label)
        to_keep.append(label)
    
    name_list = open("Dataset.csv", "w", newline='')
    name_writer = csv.writer(name_list)

    # Copiar os arquivos para o output_folder
    for src_path in to_keep:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)
        shutil.copy2(src_path, dst_path)
        src_path = str(src_path).replace('./Data\\', '')
        src_path = str(src_path).replace('.csv', '')
        name_writer.writerow([src_path])

    

    print(f"{len(to_keep)} arquivos foram copiados para {output_folder}.")
    return to_keep

folder_entrada = r"./Data"
folder_saida = r"./Data_sample"

filter_similar_csvs(folder_entrada, folder_saida, similarity_threshold=float(1-1e-4))
