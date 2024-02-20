# SCMEA

## Test

If you want to experience the alignment performance of the model directly, you can now load the SCMEA model we uploaded.

- Please download the SCMEA model from [Google Cloud Drive](https://drive.google.com/drive/folders/1jQ95KmZzopDxDT4lo3s7caGCIxyh60YN?usp=drive_link).
- Please move the npy file into `data/DBP15K/` and move the pth file into `save_pkl`

```shell
python src/test.py
```

## Train

- Please modify the hyperparameters in `parse_options` as needed.

```shell
python src/run.py
```

Requirements:

- python 	3.9.0
- torch	1.12.0
- tqdm	4.46.1
