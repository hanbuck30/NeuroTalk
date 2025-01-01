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
## 4. Sequence of Implementation
### Step 1: Modify Directories
Update DataDir and TmpDir in the startup script:
```bash
startup_bbci_toolbox('DataDir', "C:\Users\user\bbci_toolbox", 'TmpDir','/tmp/');
```
### Step 2: Vector Embedding
Run the vector embedding process:
```bash
vector_embedding
```
### Step 3: Save Results
Save the processed data into a .mat file:
```bash
save('C:\Users\user\OneDrive\바탕 화면\BXAI\matlab_code\result\CMJ\epoch_data.mat', 'fv_te_spoken', 'fv_te_imagined', 'fv_tr_imagined', 'fv_tr_spoken', 'fv_val_imagined', 'fv_val_spoken');
```
## Variance(Segmentation)
In the vector_embedding code, use the following function for variance processing:
```bash
proc_variance(fv_tr_EEG, n_sess);
```
The proc_multicsp function remains unchanged.

## Variance(Sliding Window)
In the vector_embedding code, use the following function for sliding variance processing:
```bash
proc_sliding_variance(fv_tr_EEG, window_len, n_sess, 0);
```
Again, the proc_multicsp function remains unchanged.

## Notes:

- Ensure all necessary dependencies are in place.

- Modify paths and variable names to suit your environment.

- For further customizations, refer to the bbci_toolbox documentation.


