import h5py

def inspect_hdf5_file(file_path):
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as file:
        print("Inspecting HDF5 file: ", file_path)
        
        def print_items(name, obj):
            # Determine the type of the object (group or dataset)
            obj_type = "Attributes" if isinstance(obj, h5py.Group) else "Dataset"
            print(f"\n{obj_type}: {name}")
            
            # Print attributes of the group or dataset
            if obj.attrs:
                for attr_name, attr_value in obj.attrs.items():
                    print(f"Attribute - {attr_name}: {attr_value}")
            
            # Additional details for datasets
            if isinstance(obj, h5py.Dataset):
                print(f"Shape: {obj.shape}, Type: {obj.dtype}")
                # Example to print compression info; you can add more properties to inspect
                print(f"Compression: {obj.compression}, Compression opts: {obj.compression_opts}")
                # Print a small part of the dataset's data, if it's not too large
                if obj.size <= 100:  # Adjust this threshold as needed
                    print(f"Data: {obj[:]}")
        
        # Use the visititems method to iterate through groups and datasets
        file.visititems(print_items)


file_path = r"Z:\members\Rauscher\data\big_data_small\Drosophila_PIStage_3.00mm_20240205_172031\Drosophila_PIStage_3.00mm_20240205_172031.HDF5"
inspect_hdf5_file(file_path)
