import glob
import sys
# import os


for subpackage in glob.glob("Imported_Code"):
    print(f"Adding \'{subpackage}\' to system path")
    sys.path.append(subpackage)
