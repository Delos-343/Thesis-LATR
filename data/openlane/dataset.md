# OpenLane Dataset (lane3d_1000_training.tar)

### Google Drive:
https://drive.google.com/drive/folders/18upnDfB-VVuQf3GPiv_JQn1-BUOcAotk?usp=sharing

### Baidu Cloud (Paste the URL directly into your browser):
https://pan.baidu.com/s/14Mbo1u6VndFl6O7aeEW9SQ?pwd=t6h6

### OpenDateLab: 
https://opendatalab.org.cn/OpenLane


### In root directory (LATR), if directory does not exist yet:
```
cd data
mkdir openlane
cd openlane
```

**Replace ${OPENLANE_PATH} with actual path for images and lane3d_1000**

**Don't forget to unzip images.tar and lane3d_1000_training.tar**

### Windows Powershell (Recommended)
```
mklink /D "C:\Users\moham\Desktop\PYT\LATR\data\openlane\images" "C:\path\to\OPENLANE_PATH\images"
mklink /D "C:\Users\moham\Desktop\PYT\LATR\data\openlane\lane3d_1000_training" "C:\path\to\OPENLANE_PATH\lane3d_1000"
```

### CMD
```
mklink /D images "C:\path\to\OPENLANE_PATH\images"
mklink /D lane3d_1000 "C:\path\to\OPENLANE_PATH\lane3d_1000"
```
