#  Prequel
## Description
Prequel is a music generator using Long Short Term Memory cells in a recurrent neural network (example of music can be find  [here](https://soundcloud.com/user-595610886/prequel-generated-music))
## Installation
```bash
git clone https://github.com/Slownite/Prequel.git
docker build -t prequelanaconda .  
docker run -p 3000:3000 -it prequelanaconda python3 api.py
```
## Usage
```bash
python client/client.py
```
