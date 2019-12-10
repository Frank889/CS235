What are the files in the zip file:
1. CNN_submission.py and ANN_submission.py: source python script running in Docker.
2. CNN_submission.ipynb and ANN_submission.ipynb: source jupyter notebook
3. odors_16.npy and sensory_outs_16.npy: data files
4. result: the folder which contains all the results


Docker Environment:
1. Download the Docker. You could find it at the link: https://www.docker.com/. 
2. After installing the Docker, open it.
3. Download the project environment at the link: https://hub.docker.com/r/swang277/cs235. You could also use the terminal and enter the following command: 
$ docker pull swang277/cs235


How to run our project in Docker:
1. Download the project zip file and unzip it.
2. Open the terminal, enter the following command:
$ docker run -it -v (path):/usr/cs235_project swang277/cs235 /bin/bash
Replace the (path) with the path of the project file.
3. Enter the following command to run the CNN and ANN code:
$ python CNN_submission.py ANN_submission.py


Result checking:
1. Way1: Open the Jupyter Notebook and upload the following files:
CNN_submission.ipynb , ANN_submission.ipynb
2. Way2: Open the folder named ‘result’ and check all the results.