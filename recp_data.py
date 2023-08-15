import numpy as np
import random
import torch
import torch.nn as nn

import pickle

FType = torch.FloatTensor
LType = torch.LongTensor

poi_info_path = "dataset/poi_info.pickle" 
a_path = 'dataset/attribute_m.pickle'
s_path = 'dataset/source_matrix.pickle'
d_path = 'dataset/destina_matrix.pickle'


class ReData:

    def __init__(self):
        self.a_m = pickle.load(open(a_path, "rb"))
        self.s_m = pickle.load(open(s_path, "rb"))
        self.d_m = pickle.load(open(d_path, "rb"))
        self.poi_info = pickle.load(open(poi_info_path, "rb"))


    def gaussian_noise(self, matrix, seed, mean=0, sigma=0.03, ):
        np.random.seed(seed)
        matrix = matrix.copy()
        noise = np.random.normal(mean, sigma, matrix.shape)
        mask_overflow_upper = matrix + noise >= 1.0
        mask_overflow_lower = matrix + noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        matrix += noise
        return matrix

    def add_aug(self, poi_set, seed, _ratio):
        random.seed(seed)

        add_poi_set = []
        for poi in poi_set:
            add_poi_set.append(poi)
            ratio = random.random()
            if ratio < _ratio:
                add_poi_set.append(poi)
        return add_poi_set

    def delete_aug(self, poi_set, seed, _ratio):
        random.seed(seed)

        de_poi_set = []
        for poi in poi_set:
            ratio = random.random()
            if ratio > _ratio:
                de_poi_set.append(poi)
        if not de_poi_set:
            de_poi_set = [poi_set[0]]
        return de_poi_set

    def replace_aug(self, poi_set, seed, _ratio):
        random.seed(seed)

        replace_poi_set = []
        for poi in poi_set:
            new_poi = poi
            ratio = random.random()
            if ratio < _ratio:
                new_poi[1] = random.randint(0, 250)
            replace_poi_set.append(new_poi)
        return replace_poi_set

    def get_aug(self, seed=42):
        """Augmentation function that produces positive samples."""
        poi_augs, flowup_augs, flowoff_augs = [], [], []
        for i in range(3):
            poi_train = []
            for idx in range(270):
                poi_set = self.poi_info[idx]

                ratio = 0.1
                if i == 0:
                    aug_poi_set = self.add_aug(poi_set, seed + i, ratio)
                elif i == 1:
                    aug_poi_set = self.delete_aug(poi_set, seed + i, ratio)
                else:
                    aug_poi_set = self.replace_aug(poi_set, seed + i, ratio)

                poi_f = np.zeros(251)
                for poi in aug_poi_set:
                    poi_id = poi[1]
                    poi_f[poi_id] += 1

                poi_train.append(poi_f)

            poi_train = np.array(poi_train)
            col_index = [21, 88, 108, 142, 190, 191, 210]
            poi_train = np.delete(poi_train, col_index, axis=1)

            poi_augs.append(poi_train)

        for i in range(4):
            flow_pickup, flow_dropoff = [], []
            for idx in range(270):
                pickup_matrix = self.s_m[idx]
                dropoff_matrix = self.d_m[idx]

                pickup_matrix = self.gaussian_noise(pickup_matrix, seed + i, sigma=0.0001)
                dropoff_matrix = self.gaussian_noise(dropoff_matrix, seed + i, sigma=0.0001)

                flow_pickup.append(pickup_matrix)
                flow_dropoff.append(dropoff_matrix)

            flowup_augs.append(flow_pickup)
            flowoff_augs.append(flow_dropoff)

        return [[np.array(poi_augs[0]), np.array(poi_augs[1]), np.array(poi_augs[2])],
                [np.array(flowup_augs[0]), np.array(flowup_augs[1]), np.array(flowup_augs[2]),
                 np.array(flowup_augs[3]), ],
                [np.array(flowoff_augs[0]), np.array(flowoff_augs[1]), np.array(flowoff_augs[2]),
                 np.array(flowoff_augs[3]), ],
                ]


