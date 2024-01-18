import os
import numpy as np
from utils import get_config

config = get_config()

def load_slice(slice_dir: str, slice_num: int) -> np.ndarray:
    slice_name = f"{slice_num:03}"
    slice_path = os.path.join(slice_dir, slice_name)
    
    return np.load(slice_path)


def load_volume(volume_dir, dim: int) -> np.ndarray:
    slice_dir = os.path.join(volume_dir, str(dim))
    list_slice = []
    for slice_num in range(300):
        list_slice.append(
            load_slice(slice_dir, slice_num)
        )
    
    return np.stack(list_slice)


def init_instance_volume(volume: np.ndarray) -> np.ndarray:
    id_instance = 0
    for i in range(volume.shape[0]):
        instance_slice = volume[i, :, :, 1]
        new_instance_slice = np.zeros(instance_slice.shape, dtype=np.float16)
        for id in np.unique(instance_slice):
            new_instance_slice = np.where(instance_slice == id, id_instance, new_instance_slice)
            id_instance += 1
        
        volume[i, :, :, 1] = new_instance_slice
    
    return volume, id_instance

    
def create_id_mapper(sub_slice, sub_ref_slice, nb_mask):
    # Fusionner les deux tableaux en une seule matrice 2D
    tableau_combine = np.column_stack((sub_ref_slice, sub_slice))

    # Utiliser numpy.unique pour obtenir les valeurs uniques et leurs occurrences
    tuples_uniques, tuples_occurrences = np.unique(tableau_combine, axis=0, return_counts=True)
    valeurs_uniques, occurrences = np.unique(sub_ref_slice, return_counts=True)

    # Créer le tableau 2D avec les tuples uniques et les taux de répartition
    tuples_occurrences = tuples_occurrences.astype(np.float64)

    # Utiliser np.searchsorted pour obtenir les indices des valeurs uniques dans valeurs_uniques
    indices_occurrences = np.searchsorted(valeurs_uniques, tuples_uniques[:, 0])
    # Mettre à jour le taux de répartition avec les valeurs correspondantes
    tuples_occurrences /= occurrences[indices_occurrences]

    # Créer le dictionnaire final
    id_mapper = {}
    for val in valeurs_uniques:
        rates = np.zeros(nb_mask)
        indices_val = np.where(tuples_uniques[:, 0] == val)
        rates[tuples_uniques[indices_val, 1]] = tuples_occurrences[indices_val]
        id_mapper[val] = rates

    return id_mapper


def assign_group(list_id_mapper: list[dict], nb_mask):
    unique_keys = set(key for d in list_id_mapper for key in d)
    list_group = []
    for key in unique_keys:
        group = np.full(nb_mask, 0, dtype=np.float16)
        for id_mapper in list_id_mapper:
            if id_mapper.__contains__(key):
                group += id_mapper[key]
        
        list_group.append(group)
    
    return np.stack(list_group)


def create_masks_groups(volume: np.ndarray, volume_dir, dim: int) -> np.ndarray:
    # Create a unique id for each masks of the volume
    volume, nb_mask = init_instance_volume(volume)
    # Path du volume permettant de lier chaque masque
    ref_slice_dir = os.path.join(volume_dir, dim=str(1 - dim))
    
    # matrix_group = np.zeros((volume.shape[1], nb_mask), np.int32)
    list_mask_groups = []
    for ref_idx in range(volume.shape[1]):
        ref_slice = load_slice(ref_slice_dir, ref_idx)
        list_id_mapper = []
        for idx in range(volume.shape[0]):
                list_id_mapper.append(
                    create_id_mapper(volume[idx, ref_idx, :, 1], ref_slice[idx, :, 1], nb_mask)
                )
        list_mask_groups.append(
            assign_group(list_id_mapper)
        )
        
    return list_mask_groups


def unifie_mask_groups(list_mask_groups):
    for mask_group in list_mask_groups:
        
    return 
# Je suppose que mon volume fait X Y Z C
# X = 300 : axes des X
# Y = 300 : axes des Y
# C = 2 : probabilité binaire, instance id
def merge_volumes(run_name, volume_name):
    volume_dir = os.path.join(
        get_config['submissions']['root'],
        run_name,
        volume_name
    )
    
    x_volume = load_volume(volume_dir, dim=0)
    # Création de groupes de masque pour chaque slide de référence
    list_mask_groups = create_masks_groups(x_volume, volume_dir, dim=0)
    # Algorithme permettant de passer de goupes de masque par slice à un set de groupe de masque 
    # répertoriant de manière unique chaque id d'instance
    masks_groups_unified = unifie_mask_groups(list_mask_groups)
    
    y_volume = load_volume(volume_dir, dim=1)
    y_volume = np.moveaxis(y_volume, 1, 0)
    masks_groups = create_masks_groups(y_volume, volume_dir, dim=1)
    y_volume = np.moveaxis(y_volume, 0, 1)