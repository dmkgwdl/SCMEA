# SCMEA


## Test
If you want to experience the alignment performance of the model directly, you can now load the SCMEA model we uploaded.
- Please download the pre-trained word embedding from [Google Cloud Drive](https://drive.google.com/drive/folders/1j76JB_cbeZ74VU2p3O3EervZjkhWOvhm?usp=drive_link).
- Please download the SCMEA model from [Google Cloud Drive](https://drive.google.com/drive/folders/1jQ95KmZzopDxDT4lo3s7caGCIxyh60YN?usp=drive_link).
- Please move the npy file into `data/DBP15K/` and move the pth file into `save_pkl`
```shell
python src/test.py
```
## Train
- Please modify the hyperparameters in `parse_options` as needed.
```shell
CUDA_VISIBLE_DEVICES='-1' python src/run.py
```
