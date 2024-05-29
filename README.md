This MPI calculation is based on pyPI written by Daniel, Gillford. 
This code run on Linux, VScode. 
First, you need to install some extensions and create conda environment. 

Extension : Jupyter, Jupyter Cell Tags, Jupyter Slide Show, Jupyter Keymap, Python, Pylance, Python extended, vscode-pdf

conda environment : There is requirements.txt about packeges needed. Create conda environment, check packeges and then install in it.

Then in terminal, command 
'git clone 'url of this repository''

/Python_Earthscience/MPI_ex/MPI_mixing_depth_temp/mpi_preTC.py 
: Calculate pre MPI
/Python_Earthscience/MPI_ex/MPI_mixing_depth_temp/mpi_durTC.py
: Calculate dur MPI 

If you see mpi_preTC.py or mpi_durTC.py, there are '#%%' in code. 
This is cell of Python and if you command 'shift+enter', then jupyter file will be created. 
Before you command, check python interpreter that yours conda environment is selected correctly. 


Python_Earthscience/MPI_ex/data : Input, output data will be stored in this directory
Python_Earthscience/MPI_ex/figure : Some figures will be stored in this directory
Python_Earthscience/MPI_ex/tcpyPI-1.3 : This file is original pyPI, but it's unnecessary because I changed code for this modules, so you don't need to use this file. If you want to see whole code of pyPI, then check it. 
Python_Earthscience/MPI_ex/wind_stress : This file is about calculating wind stress example. In calculating durTC MPI, there is mixing temperature. I changed wind stress to newer version for mixing temperature. 
