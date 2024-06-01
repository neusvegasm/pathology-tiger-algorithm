import gcapi
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import os


# authorise with your personal token
my_personal_GC_API_token = '0efc24961ab263566ba64ff87579a2079cffc932b366a977b96976f60ca4a385'
client = gcapi.Client(token=my_personal_GC_API_token)
print(client.algorithms)

# retrieve the algorithm, providing a slug
algorithm_1 = client.algorithms.detail(slug="nnunet_segmentation_for_detection")

# explore, which input the algorithm expects
algorithm_1["inputs"]

