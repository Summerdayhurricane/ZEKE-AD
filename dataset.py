import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os


class MRIADDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]

		if mode == 'train':
			meta_info = meta_info['train']
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name == None :
				self.cls_names = list(meta_info.keys())
			else:
				self.cls_names = [obj_name]
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				interval_size = 60
				total_numbers = len(data_tmp)
				intervals = (total_numbers + interval_size - 1) // interval_size  # Ceiling division
				res_k_shot = k_shot
				# indices = torch.randint(0, len(data_tmp), (k_shot,))
				# List to store results
				indices = []
				# Generate random numbers for each interval
				for i in range(intervals):
					start = i * interval_size
					end = min(start + interval_size, total_numbers)  # Ensure it does not exceed the total
					interval_range = end - start  # Length of current interval
					count = ((interval_size * res_k_shot) // (total_numbers-start)) if i < intervals - 1 else res_k_shot  # Last interval fills the remaining random numbers
					res_k_shot = res_k_shot - count
					# Generate non-repeating random numbers
					random_indices = torch.randperm(interval_range)[:count] + start
					indices.append(random_indices)
				# Merge random numbers from all intervals into one tensor
				indices = torch.cat(indices)
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info["test"][cls_name])
				self.data_all.extend(meta_info["train"][cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			img = Image.open(img_path)
			img_ls.append(img)
		# Image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		return result_image

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, cls_name, specie_name = data['img_path'], data['cls_name'], data['specie_name']
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
		# Image transform
		img = self.transform(img) if self.transform is not None else img
		return {'img': img, 'cls_name': cls_name, 'img_path': os.path.join(self.root, img_path)}

	def remove_train_data_from_test(self, train_data):
		# Get all img_path from train_data
		train_img_paths = set()
		for data in train_data.data_all:
			train_img_paths.add(data['img_path'])

		# Remove samples appearing in train_data from test_data
		filtered_test_data = [data for data in self.data_all if data['img_path'] not in train_img_paths]

		# Update data_all and length of test_data
		self.data_all = filtered_test_data
		self.length = len(filtered_test_data)
