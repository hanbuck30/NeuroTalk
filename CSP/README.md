# CSP
This document outlines the steps to implement CSP using the bbci_toolbox framework. It includes guidelines on preprocessing raw data, handling errors, and performing operations like variance calculation and sliding window processing.

## 1. File Adjustments
### Move Functions
- Ensure proc_variance and proc_sliding_variance are located within the bbci_public/processing/ directory.

## 2. Check Raw Data's Columns
- Concatenate the following columns in your raw data: "time", "Data", "Label".
- Alternatively, modify the CSP code to fit your data structure.

## 3. Handling Parport Errors
- The bbci_toolbox requires some modifications for smooth execution. Ensure the code is adjusted, and files are placed in the correct path:
  ```bash matlab
  bbci_public/
  ```
