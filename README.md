# Project from Hackathon
## Environment preparation:
First install Anaconda on your computer from: https://www.anaconda.com/download you can keep all default options for all the configurations of the Installation.
Later open Anaconda Prompt and run the following commands:
Conda name can be modified as well as the Python version.
```
conda create --name myenv python=3.8
conda activate myenv  
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install transformers
pip install pandas
pip install transformers[torch]
```

Installation could take a couple of minutes.
Now you can start using Python.
```
conda activate myenv  
python
```
* If the installation was successful, you can see the version of Python.
* Then download the AOP-ALL file (.CSV) and change the path of the file in the Python code according to the location of the file in your computer.
* Download the dataset01 file (.CSV) as the validation dataset.
* Copy the Python code (T5) and run the command.
