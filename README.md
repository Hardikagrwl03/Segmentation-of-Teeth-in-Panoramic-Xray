# Semantic-Segmentation-of-Teeth-in-Panoramic-X-ray-Image

The aim of this study is automatic semantic segmentation and measurement total length of teeth in one-shot panoramic x-ray image by using deep learning method with U-Net Model and binary image analysis in order to provide diagnostic information for the management of dental disorders, diseases, and conditions. 

*****Architecture.*****

<img src="https://github.com/Hardikagrwl03/Segmentation-of-Teeth-in-Panoramic-Xray/blob/master/Viewing_Estimations/gallery/Architecture.png" alt="Results" width="1024" height="512">

###  Getting Started

Initially clone the repository and download necessary dependencies using a Python 3.7 environment.

```bash
git clone git@github.com:Hardikagrwl03/Segmentation-of-Teeth-in-Panoramic-Xray.git
pip install -r requirements.txt
```
Now the dataset must be imported using the following commands:

```bash
python download_dataset.py
python mask_prepare.py
```
To directly semantically segment the Xray Image, use the following command:
```bash
python main.py -p <Path of Xray Image>
```
To train the model as well include a -t along with the command:
```bash
python main.py -t -p <Path of Xray Image>
```

### Sample Output

<!-- <img src="https://github.com/Hardikagrwl03/Segmentation-of-Teeth-in-Panoramic-Xray/gallery/Sample_Output.png" alt="Results" width="1024" height="512"> -->

### Original Dataset

DATASET ref - 	H. Abdi, S. Kasaei, and M. Mehdizadeh, “Automatic segmentation of mandible in panoramic x-ray,” J. Med. Imaging, vol. 2, no. 4, p. 44003, 2015

[Link DATASET for only original images.](https://data.mendeley.com/datasets/hxt48yk462/1)

### Paper  

[The authors of this article are Selahattin Serdar Helli and Andaç Hamamcı  with the Department of Biomedical Engineering, Faculty of Engineering, Yeditepe University, Istanbul, Turkey](https://dergipark.org.tr/tr/pub/dubited/issue/68307/950568) 

### BibTeX Entry and Citation Info
 ```
@article{helli10tooth,
  title={Tooth Instance Segmentation on Panoramic Dental Radiographs Using U-Nets and Morphological Processing},
  author={HELL{\.I}, Serdar and HAMAMCI, Anda{\c{c}}},
  journal={D{\"u}zce {\"U}niversitesi Bilim ve Teknoloji Dergisi},
  volume={10},
  number={1},
  pages={39--50}
}
 ```
