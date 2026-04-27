# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import glob
import csv
import logging
import re
import torch
import xml.dom.minidom as XD
import os.path as osp
import xml.etree.ElementTree as ET

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from utils.class_aware import add_class_metadata, build_class_to_idx, read_class_csv
@DATASET_REGISTRY.register()


class UrbanElementsReID(ImageDataset):

    def __init__(self, root='/home/muqsitamir/datasets/Urban2026/',
                 verbose=True, **kwargs):
        self.class_aware = kwargs.pop('class_aware', False)
        self.class_aware_cfg = kwargs.pop('class_aware_cfg', None)
        self.dataset_dir = root

        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'image_train/')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_train/')

        self._check_before_run()
        self.class_to_idx = self._build_class_to_idx()
        self.num_semantic_classes = len(self.class_to_idx)
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(UrbanElementsReID, self).__init__(self.train , self.query, self.gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _readCSV_(self, csv_dir):
        
        camids = []
        imageNames = []
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(int(row[2]))
        
        return list(zip(camids, imageNames, pids))
    
    def _readCSV_eval_(self, csv_dir):
        camids = []
        imageNames = []
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(-1)
        
        return list(zip(camids, imageNames, pids))

    def _class_cfg_value(self, key, default):
        if self.class_aware_cfg is None:
            return default
        return getattr(self.class_aware_cfg, key, default)

    def _build_class_to_idx(self):
        if not self.class_aware:
            return {}
        train_csv = self._class_cfg_value('TRAIN_CSV', 'train_classes.csv')
        return build_class_to_idx(self.dataset_dir, train_csv)

    def _read_class_map(self, csv_name):
        if not self.class_aware:
            return {}
        return read_class_csv(osp.join(self.dataset_dir, csv_name))
    
    def _process_dir(self, dir_path, relabel=False):
        filt_path = osp.join(self.dataset_dir, 'train_filtered.csv')
        xml_dir = filt_path if osp.exists(filt_path) else osp.join(self.dataset_dir, 'train.csv')
        default_classes = 'train_classes_filtered.csv' if xml_dir.endswith('train_filtered.csv') else 'train_classes.csv'
        class_csv = self._class_cfg_value('TRAIN_CSV', default_classes)
        logging.getLogger('PAT').info("UrbanElementsReID: loading train CSV: %s  classes CSV: %s", xml_dir, class_csv)
        xml_file = self._readCSV_(xml_dir)
        image_to_class = self._read_class_map(class_csv)

        pid_container = set()

        for _, _, pid in xml_file:
            if pid == -1: continue
            pid_container.add(pid)       
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []

        for camid, imageName, pid in xml_file:
            camid = int(camid[1:])
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            item = (osp.join(dir_path, imageName), pid, camid)
            if self.class_aware:
                item = add_class_metadata(item, imageName, image_to_class, self.class_to_idx)
            dataset.append(item)
                
        return dataset
    
    def _process_dir_test(self, dir_path, relabel=False, query=True):
        
        dataset = []

        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query.csv')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test.csv')

        xml_file = self._readCSV_eval_(xml_dir)
        
        for camid, imageName, pid in xml_file:
            camid = int(camid[1:])
            dataset.append((osp.join(dir_path, imageName), pid, camid))
                
        return dataset 

    def _process_track(self, path): 
        
        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            tracklet[track_id] =  curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
                
        return tracklet, nums, frame2trackID

    def _process_dir_testVeri(self, dir_path, relabel=False):
        
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')
        pid_container = set()
        
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
