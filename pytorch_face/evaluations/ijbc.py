import time
import os
import cv2
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from scipy import interpolate
import torch

# pool_func
def average_then_normalize(features):
    features = np.mean(features, axis=0)
    features = features / np.linalg.norm(features)
    return features

def normalize_then_average(features):
    features = features / np.linalg.norm(features, axis=1)[:, None]
    features = np.mean(features, axis=0)
    return features

# agg_func
def softmax(pairwise_distances): 
    pass

# transform_func
def normalize(feature):
    return feature / np.linalg.norm(feature)

def _read_ijbc_verification_template(metadata_dir):
    gallery_G1_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_gallery_G1.csv")
    gallery_G2_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_gallery_G2.csv")
    probe_mixed_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_probe_mixed.csv")
    
    template = {}
    with open(gallery_G1_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            template[template_id] = subject_id 

    with open(gallery_G2_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            template[template_id] = subject_id 

    with open(probe_mixed_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            template[template_id] = subject_id 

    return template

def _read_ijbc_verification_pair(metadata_dir):
    pair_file_path = os.path.join(metadata_dir, "ijbc/ijbc_11_G1_G2_matches.csv")

    pairs = []
    with open(pair_file_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            pair = [i for i in tokens]
            pairs.append(pair)

    return pairs

def evaluate_verification(model, transforms, data_dir, metadata_dir, far_levels, normalize_func=None, pool_func=None, agg_func=None):
    print("+------------------------------------------------------------+")
    print("| EVALUATING MODEL VERIFICATION PERFORMANCE ON IJB-C DATASET |")
    print("+------------------------------------------------------------+")
    print("==> reading evaluation configurations")
    if normalize_func:
        print(" - {0:<16}: {1}".format("transform_func", normalize_func.__name__))
    if pool_func:
        print(" - {0:<16}: {1}".format("pool_func", pool_func.__name__))
    if agg_func:
        print(" - {0:<16}: {1}".format("agg_func", agg_func.__name__))
    
    print("==> reading test metadata")
    pairs = _read_ijbc_verification_pair(metadata_dir)
    template_metadata = _read_ijbc_verification_template(metadata_dir)

    template_features = {}
    for template_id in tqdm(os.listdir(data_dir), desc="==> reading features", ncols=0):
        template_dir = os.path.join(data_dir, template_id)
        features = []
        for file_name in os.listdir(template_dir):
            if file_name.endswith(".jpg"):
                img_path = os.path.join(template_dir, file_name)
                img = np.float32(cv2.imread(img_path))
                input = torch.cat((transforms(img).unsqueeze_(0), transforms(np.fliplr(img)).unsqueeze_(0)))
                output = model(input).detach().to('cpu').numpy()
                feature = np.hstack([output[0], output[1]])
                if normalize_func:
                    feature = normalize_func(feature)
                features.append(feature)
        if features:
            features = np.array(features)
            if pool_func:
                features = pool_func(features)
            template_features[template_id] = features
    model.eval()
    pair_types = []
    similarity_scores = []
    for pair in tqdm(pairs, desc="==> computing similarity scores", ncols=0):
        if pair[0] not in template_features or pair[1] not in template_features:
            continue

        pair_types.append(int(template_metadata[pair[0]] == template_metadata[pair[1]]))

        left_features = template_features[pair[0]]
        right_features = template_features[pair[1]]

        similarity_score = left_features.dot(right_features.T)
        if agg_func:
            similarity_score = agg_func(similarity_score)

        if similarity_score.ndim != 0:
            raise ValueError("agg_func provided is not valid")
        similarity_scores.append(similarity_score)

    pair_types = np.array(pair_types)
    similarity_scores = np.array(similarity_scores)

    far, tar, thresholds = metrics.roc_curve(pair_types, similarity_scores)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]

    print("[Verification Score]")
    for (far, tar) in zip(far_levels, tar_at_far):
        print(" - TAR@FAR={0:.1e}: {1:.4f}".format(far, tar))

    return tar_at_far
    

def _read_ijbc_identification_gallery(metadata_dir):
    gallery_G1_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_gallery_G1.csv")
    gallery_G2_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_gallery_G2.csv")

    gallery_metadata = {}
    with open(gallery_G1_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            if subject_id not in gallery_metadata:
                gallery_metadata[subject_id] = []
            gallery_metadata[subject_id].append(template_id)

    with open(gallery_G2_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            if subject_id not in gallery_metadata:
                gallery_metadata[subject_id] = []  
            gallery_metadata[subject_id].append(template_id)

    return gallery_metadata


def _read_ijbc_identification_probe(metadata_dir):
    probe_mixed_file_path = os.path.join(metadata_dir, "ijbc/ijbc_1N_probe_mixed.csv")
    probe_metadata = {}
    with open(probe_mixed_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')    
            template_id = tokens[0]
            subject_id = tokens[1]
            probe_metadata[template_id] = subject_id 
    return probe_metadata

def evaluate_identification(model, transforms, data_dir, metadata_dir, rank_levels, fpir_levels, normalize_func=None, pool_func=None, agg_func=None):
    print("+--------------------------------------------------------------+")
    print("| EVALUATING MODEL IDENTIFICATION PERFORMANCE ON IJB-C DATASET |")
    print("+--------------------------------------------------------------+")
    print("==> reading evaluation configurations")
    if normalize_func:
        print(" - {0:<16}: {1}".format("transform_func", normalize_func.__name__))
    if pool_func:
        print(" - {0:<16}: {1}".format("pool_func", pool_func.__name__))
    if agg_func:
        print(" - {0:<16}: {1}".format("agg_func", agg_func.__name__))
        
    gallery_metadata = _read_ijbc_identification_gallery(metadata_dir)
    probe_metadata = _read_ijbc_identification_probe(metadata_dir)

    gallery_subject_ids = []
    gallery_templates = []
    for subject_id in tqdm(gallery_metadata, desc="==> reading gallery features", ncols=0):
        features = []
        template_ids = gallery_metadata[subject_id]
        for template_id in template_ids:
            template_dir = os.path.join(data_dir, template_id)
            if not os.path.exists(template_dir):
                continue
            for file_name in os.listdir(template_dir):
                if file_name.endswith(".jpg"):
                    img_path = os.path.join(template_dir, file_name)
                    img = np.float32(cv2.imread(img_path))
                    input = torch.cat((transforms(img).unsqueeze_(0), transforms(np.fliplr(img)).unsqueeze_(0)))
                    output = model(input).detach().to('cpu').numpy()
                    feature = np.hstack([output[0], output[1]])
                    if normalize_func:
                        feature = normalize_func(feature)
                    features.append(feature)
        if features:
            features = np.array(features)
            if pool_func:
                features = pool_func(features)
            gallery_subject_ids.append(subject_id)
            gallery_templates.append(features)
    gallery_subject_ids = np.array(gallery_subject_ids)
    gallery_templates = np.array(gallery_templates)

    probe_subject_ids = []
    probe_templates = []
    for template_id in tqdm(probe_metadata, desc="==> reading probe features", ncols=0):
        features = []
        subject_id = probe_metadata[template_id]
        if subject_id not in gallery_subject_ids:
            continue
        template_dir = os.path.join(data_dir, template_id)
        if not os.path.exists(template_dir):
            continue
        for file_name in os.listdir(template_dir):
            if file_name.endswith(".jpg"):
                img_path = os.path.join(template_dir, file_name)
                img = np.float32(cv2.imread(img_path))
                input = torch.cat((transforms(img).unsqueeze_(0), transforms(np.fliplr(img)).unsqueeze_(0)))
                output = model(input).detach().to('cpu').numpy()
                feature = np.hstack([output[0], output[1]])
                if normalize_func:
                    feature = normalize_func(feature)
                features.append(feature)
        if features:
            features = np.array(features)
            if pool_func:
                features = pool_func(features)
            probe_subject_ids.append(subject_id)
            probe_templates.append(features)
    probe_subject_ids = np.array(probe_subject_ids)
    probe_templates = np.array(probe_templates)

    model.eval()
    thresholds = np.arange(-1, 1, 0.001)
    tps = np.zeros(len(thresholds))
    fps = np.zeros(len(thresholds))
    similarity_scores = np.zeros((len(probe_templates), len(gallery_templates)))
    for i in tqdm(range(len(probe_templates)), desc=" - computing similarity scores", ncols=0):
        for j in range(len(gallery_templates)):    
            similarity_score = probe_templates[i].dot(gallery_templates[j].T)
            if agg_func:
                similarity_score = agg_func(similarity_score)
            if similarity_score.ndim != 0:
                raise ValueError("agg_func provided is not valid")
            similarity_scores[i, j] = similarity_score

            if probe_subject_ids[i] == gallery_subject_ids[j]:
                tps += similarity_score > thresholds
            else:
                fps += similarity_score > thresholds 
    sorted_score_idxs = np.argsort(similarity_scores)[:, ::-1]
    sorted_predictions = np.array(gallery_subject_ids).take(sorted_score_idxs)
    argmatch = np.where(sorted_predictions == np.array(probe_subject_ids)[:, None])[1]
    cmc = np.zeros(similarity_scores.shape)
    for i in range(len(probe_templates)):
        cmc[i, argmatch[i]:] = 1
    cmc = np.mean(cmc, axis=0)
    cmc = cmc[rank_levels]
    print("[CMC Score]")
    for (rank, accuracy) in zip(rank_levels, cmc):
        print(" - Accuracy@Rank={0:d}: {1:.4f}".format(rank, accuracy))

    tpir = tps / len(probe_templates)
    fpir = fps / len(probe_templates)
    interp = interpolate.interp1d(fpir, tpir)
    tpir_at_fpir = [interp(x) for x in fpir_levels]

    print("[DET Score]")
    for (fpir, tpir) in zip(fpir_levels, tpir_at_fpir):
        print(" - TPIR@FPIR={0:.1e}: {1:.4f}".format(fpir, tpir))
    
    return cmc, tpir_at_fpir

