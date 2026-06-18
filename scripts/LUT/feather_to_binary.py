

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

from pathlib import Path


if __name__ == "__main__":
    
    # Load your feather file


    # Load feather
    df = pd.read_feather(Path(__file__).parent / "vmf_lut.feather")

    print(df)

    # Convert floating point columns to 0-based integer indices safely
    # uint16 is perfect here (handles up to 65,535 indices)
    kappa_idx = df['kappa'].astype('category').cat.codes.astype(np.uint16)
    th1_idx = df['th1'].astype('category').cat.codes.astype(np.uint16)
    th2_idx = df['th2'].astype('category').cat.codes.astype(np.uint16)
    dphi_idx = df['dphi'].astype('category').cat.codes.astype(np.uint16)

    # Open binary file
    with open("vmf_lut.bin", "wb") as f:
        # 1. Write the total number of rows (uint64)
        num_rows = np.uint64(len(df))
        f.write(num_rows.tobytes())

        # 2. Write the index arrays (uint16)
        f.write(kappa_idx.to_numpy().tobytes())
        f.write(th1_idx.to_numpy().tobytes())
        f.write(th2_idx.to_numpy().tobytes())
        f.write(dphi_idx.to_numpy().tobytes())

        # 3. Write the actual data column (float64)
        f.write(df['I'].to_numpy(dtype=np.float64).tobytes())


        # Open a binary file for writing
        # with open("data.bin", "wb") as f:
        #     # 1. Write the number of rows as an 8-byte unsigned integer (uint64)
        #     num_rows = np.uint64(len(df))
        #     f.write(num_rows.tobytes())

        #     # 2. Write the raw bytes of each column
        #     # Ensure you specify the exact C-types you expect on the C++ side
        #     f.write(df['my_doubles'].to_numpy(dtype=np.float64).tobytes())
        #     f.write(df['my_ints'].to_numpy(dtype=np.int32).tobytes())