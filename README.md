##Prequel
docker build -t prequelanaconda 
docker run -p 3000:3000 -it prequelanaconda python3 /home/api.py