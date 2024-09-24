import os

def daedalus(data_dir):
    """
    Returns dictionaries containing paths to specific file types within a given directory structure.

    Returns:
        SIMPLE (dict): Paths to 'pv_json1' and 'pv_json2' folders.
        SORTED (dict): Paths to 'sortedDomain.vtk' files.
        RAW (dict): Paths to 'vesselNetwork.vtk' files.
        RAW_UPDATED (dict): Paths to 'vesselNetwork_upDated.vtk' files.
    """

    SIMPLE = {}
    SORTED = {}
    RAW = {}
    RAW_UPDATED = {}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".vtk"):
                parent_folder = os.path.basename(os.path.dirname(root))
                full_path = os.path.join(root, file)

                if file == "sortedDomain.vtk":
                    SORTED[parent_folder] = full_path
                elif file == "vesselNetwork.vtk":
                    RAW[parent_folder] = full_path
                elif file == "vesselNetwork_upDated.vtk":
                    RAW_UPDATED[parent_folder] = full_path

        for dir in dirs:
            if dir in ["pv_json1", "pv_json2"]:
                SIMPLE[dir] = os.path.join(root, dir)

    return SIMPLE, SORTED, RAW, RAW_UPDATED

if __name__ == "__main__":
    SIMPLE, SORTED, RAW, RAW_UPDATED = daedalus()

    print("SIMPLE:")
    for key, value in SIMPLE.items():
        print(f"  {key}: {value}")

    print("\nSORTED:")
    for key, value in SORTED.items():
        print(f"  {key}: {value}")

    print("\nRAW:")
    for key, value in RAW.items():
        print(f"  {key}: {value}")

    print("\nRAW_UPDATED:")
    for key, value in RAW_UPDATED.items():
        print(f"  {key}: {value}")