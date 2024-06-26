import h5py

def list_group_contents(file_path):
    """
    Recursively lists the contents of groups within an HDF5 file,
    including subgroups and datasets.
    """
    def inspect(name, obj):
        # Identify if the current object is a group or dataset
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
        # If it's neither a group nor a dataset, it's something unexpected
        else:
            print(f"Unknown: {name}")

    with h5py.File(file_path, 'r') as hdf_file:
        # Visit every item in the HDF5 file recursively
        hdf_file.visititems(inspect)

# Replace 'your_hdf5_file.h5' with the path to your actual HDF5 file
file_path = r"Z:\members\Rauscher\data\big_data_small\Drosophila_PIStage_3.00mm_20240205_172031\Drosophila_PIStage_3.00mm_20240205_172031.HDF5"
list_group_contents(file_path)

