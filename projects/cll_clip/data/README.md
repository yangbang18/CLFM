# CLL-CLIP Data

## TOC
- [Annotations and Corpora](#annotations-and-corpora)
- [Raw Images](#raw-images)
- [Langauges](#langauges)
- [Notes for the cache folder](#notes-for-the-cache-folder)
- [Citation](#citation)


## Annotations and Corpora
Download our data from [Google Drive](https://drive.google.com/drive/folders/1FdwgjJCuGDzUVY5_38RKcNnU2_OCT6sk?usp=sharing) or [PKU Yun](https://disk.pku.edu.cn/link/AA250CD94DA8C44505B8CFD8730D7E9C89). There are three `.zip` files:
<div align="center">
<table border="1" width="70%">
    <tr align="center">
        <th>Filename</th><th>Size</th>
    </tr>
    <tr align="center">
        <td>annotations.zip</td><td>323.97 MB</td>
    </tr>
    <tr align="center">
        <td>corpus_multilingual_cc3m.zip</td><td>3.95 GB</td>
    </tr>
    <tr align="center">
        <td>corpus_multilingual_coco.zip</td><td>648.64 MB</td>
    </tr>
</table>
</div>

After downloading, directly unzip them under the data folder. The data structure should look like:
```
CLFM/projects/cll_clip/
    data
    ├── corpus
    │  ├── multilingual_coco            # MSCOCO's English training captions + Google Translator
    │  │    └── 36langs                 # covering 36 languages
    │  │        ├── cc3m_en.tsv.gz      
    │  │        ├── ...
    │  │        └── cc3m_en-zh.tsv.gz   
    │  └── multilingual_cc3m            # CC3M's English training captions + Google Translator
    │       └── 36langs                 
    │           ├── cc3m_en.tsv.gz      # 2,348,709 English captions (we drop duplicates)
    │           ├── ...
    │           ├── cc3m_en-zh.tsv.gz   # 2,348,709 English-Chinese pairs (we drop duplicates)
    │           └── line2idx.json       # 3,318,333 key-value, for undoing the dropping operation
    └── annotations                 
        ├── coco
        │   ├── en/                     # official English annotations
        │   ├── ...
        │   └── translated              # machine-translated annotations
        │       ├── ar/                 
        │       ├── ...                 # totally 36 languages, each language correspond to a folder
        │       └── zh/
        ├── cc3m
        │   └── en/                     # official English annotations
        └── xm3600
            ├── ar/                     # official Arabic annotations
            ├── ...                     # totally 36 languages, each language correspond to a folder
            └── zh/                     # official Chinese annotations
```

**Notes:** see [prepare_text_data.ipynb](/projects/cll_clip/data/prepare_text_data.ipynb) to know how we prepare the above data.



## Raw Images
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Official Link</th><th>Shared Link (Ours)</th>
    </tr>
    <tr align="center">
        <td>XM3600</td><td><a href=https://google.github.io/crossmodal-3600>Link</a></td><td>N/A</td>
    </tr>
    <tr align="center">
        <td>MSCOCO</td><td><a href=https://cocodataset.org/#download>Link</a></td><td>N/A</td>
    </tr>
    <tr align="center">
        <td>CC3M</td><td><a href=https://github.com/google-research-datasets/conceptual-captions>Link (only image urls)</a></td><td><a href=https://disk.pku.edu.cn/link/AA1484F0FFBD6A421693B152FCD6296CD0>PKU Yun</a> (~100G)</td>
    </tr>
</table>
</div>

After downloading, please organize raw images as follows (depending on [configs.image_video_root](/configs/__init__.py#L2)):
```
CLFM/projects/cll_clip/
    data
    ├── XM3600
    │   ├── 000411001ff7dd4f.jpg
    │   └── ...
    ├── MSCOCO
    │   ├── train2014
    │   │   ├── COCO_train2014_000000000009.jpg
    │   │   └── ...
    │   └── val2014
    │       ├── COCO_val2014_000000000042.jpg
    │       └── ...
    └── cc3m
        ├── images
        │   ├── 0_2901536091
        │   └── ...
        └── validation
            ├── 0_1595581236
            └── ...
```
**Note:** 
- We use [DownloadConceptualCaptions](https://github.com/igorbrigadir/DownloadConceptualCaptions) to download cc3m images, whose filenames follow the format `{LineNumber}_{HashOfImageURL}` (refer to [code](https://github.com/igorbrigadir/DownloadConceptualCaptions/blob/efb16f028936e6c628b6ee435765d6e1771b0f2d/download_data.py#L47)). In our case, the number of training and validation images of CC3M is 3,035,376 and 12,881, respectively. If you are using a different subset of CC3M (e.g., less number of images, different filename format), please modify `data/annotations/cc3m/en/*.json` accordingly.
- If you download our CC3M raw images, you will have 10 chunked files (`cc3m_train_images.tgz.0*`) for the training set. You should first run `cat cc3m_train_images.tgz.0* > cc3m_train_images.tgz` to merge them into a single file, and then run `tar -xzvf cc3m_train_images.tgz`.


## Langauges
![languages](/projects/cll_clip/asserts/languages.png)


## Notes for the cache folder
please set `CKPT_HOME` or `ZERONLG_HOME` in the environment (e.g., `export CKPT_HOME=data/checkpoints`) to specify the cache folder that stores pre-trained models. If not specified, the default cache folder will be `~/.cache/torch/zeronlg`.


## Citation
Please consider citing our paper if you use our data, thanks sincerely!

```bibtex
@inproceedings{yang2024embracing,
  title={Embracing Language Inclusivity and Diversity in CLIP through Continual Language Learning},
  author={Yang, Bang and Dai, Yong and Cheng, Xuxin and Li, Yaowei and Raza, Asif and Zou, Yuexian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6458--6466},
  year={2024}
}
```