import json
import glob
import os

def find_file_absolute_path(folder, filename):
    # Construct the search pattern
    search_pattern = os.path.join(folder, filename)

    # Use glob to find all matching files
    matching_files = glob.glob( os.path.join(folder, f"**/{filename}") )

    return None if len(matching_files)==0 else matching_files[0]

# Write now this is hard-coded for ABO. We'll check it later for objaverse

def read_abo_categories_info(text_file):
    data = {}
    glb_data = {}
    with open(text_file,'r') as f:
        for entry_ in f.readlines():
            entry_ = entry_.split('\n')[0].split(',')
            if entry_[1] not in data.keys():
                data[entry_[1]] = []
            data[entry_[1]].append(entry_[0])
            glb_data[entry_[0]] = entry_[1]
    return data, glb_data

def count_entries(category_data):
    count = 0
    for key,elem in category_data.items():
        count += len(elem)
    return count

ABO_CATEGORIES_TXT_FILE = "reflection/resources/abo_classes_3d.txt"

category_data, glb_data = read_abo_categories_info(ABO_CATEGORIES_TXT_FILE)
print(f"Sanity check. Number of data-points :{count_entries(category_data)}")

"""
ABO categories:
dict_keys(['table', 'lamp', 'bed', 'chair', 'sofa', 'dining set', 'rug', 'shelf', 'pillow', 'clock', 'picture frame or painting', 'mouse pad', 'plant or flower pot', 'cabinet', 'dresser', 'ladder', 'mirror', 'battery charger', 'ottoman', 'laptop stand', 'fan', 'instrument stand', 'exercise weight', 'container or basket', 'soap dispenser', 'electrical cable', 'exercise mat', 'vase', 'speaker stand', 'step stool', 'mount', 'air conditioner', 'cart', 'bench', 'heater', 'mattress', 'tent', 'jar', 'bag', 'shredder', 'floor mat', 'cooking pan', 'wagon', 'clothes rack', 'bowl', 'bottle rack', 'file folder', 'book or journal', 'clothes hook', 'tray', 'trash can', 'candle holder', 'holder', 'office appliance', 'birdhouse', 'drink coaster', 'cup', 'figurine or sculpture', 'sports equipment', 'vanity', 'easel', 'fire pit', 'exercise equipment'])
"""

ABO_PAIRED_DATA = [{'table', 'lamp', 'bed', 'chair', 'sofa','rug','pillow','cabinet', 'dresser','ottoman','mattress','bottle rack'},
				   {'dining set','shelf','clock', 'picture frame or painting','plant or flower pot','container or basket','soap dispenser','vase','jar','cooking pan','bowl','tray','drink coaster','cup','figurine or sculpture'},
				   {'mouse pad','battery charger','laptop stand','electrical cable','bag','shredder','file folder','book or journal','holder', 'office appliance'},
				   {'ladder','mirror','fan','instrument stand','speaker stand','step stool','mount','tent','clothes rack','clothes hook','trash can','candle holder'},
				   {'exercise weight','exercise mat','floor mat','sports equipment','exercise equipment'},
				   {'air conditioner','heater','vanity'},
				   {'cart','wagon','bench','birdhouse','easel','fire pit'} ]

CATEGPRY_ID_TO_CLUSTER_ID = {}
for c_id in category_data.keys():                
    cluster_id = 0
    for list_id,cluster in enumerate(ABO_PAIRED_DATA):
        if c_id in cluster:
            cluster_id = list_id
            break
    CATEGPRY_ID_TO_CLUSTER_ID[c_id] = cluster_id

for i,_ in enumerate(ABO_PAIRED_DATA):
    ABO_PAIRED_DATA[i] = list(ABO_PAIRED_DATA[i])

#Write this data to json
json_data = {}
json_data['category_data'] = category_data
json_data['glb_data'] = glb_data
json_data['paired_data'] = ABO_PAIRED_DATA
json_data['category_to_cluster'] = CATEGPRY_ID_TO_CLUSTER_ID

path_data = {}
for elem in json_data['glb_data'].keys():
    path_data[elem] = find_file_absolute_path("/data/manan/data/amazon_3d_data/3dmodels/original", 
                                                f"{elem}.glb")
json_data['path_data'] = path_data

find_file_absolute_path
with open("reflection/resources/abo_multi_obj.json",'w') as f:
    json.dump(json_data,f,indent=4)