# CudaAwareMPINumba
How to install and run Cuda aware MPI with Numba and send device (GPU) memory via MPI

## Installation

1. Install Ubuntu on usb [link](https://ubuntu.com/tutorials/create-a-usb-stick-on-windows#1-overview)
   1. Eject the usb once formatted
   2. Turn off the computer
   3. Insert the USB stick
   4. Hold F2 / Shift
   5. Turn on the computer
   6. In the bios menu click the drive
   7. Start Linux. If it freezes press "e" on Ubuntu and add `nomodeset` at the end of `Linux` and press Ctrl+x to continue (link)[https://itsfoss.com/fix-ubuntu-freezing/]
2. Install stuff
   1. sudo apt-get install make
   2. sudo apt-get install gcc g++
   3. sudo apt-get install python3.8
   4. sudo apt-get install pip
3. Install cuda using the debian installer (link)[https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu-x86_64-deb]
4. Update path
   1. export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
   2. export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   3. export CUDA_HOME=/usr/local/cuda-11.2
4. Install Open MPI
   1. (openmpi-4.1.0.tar.gz)[https://www.open-mpi.org/software/ompi/v4.1/]
   2. tar -xzf openmpi-4.1.0.tar.gz
   3. cd openmpi-4.1.0
   4. ./configure --with-cuda=/usr/local/cuda-11.2
   5. sudo make build install
   6. If there are failures in the process (like missing make) delete the folder unzip again and repeat
   7. At this point you should be able to run `mpicc` and `mpiexec`. If not you may need to add it to your path (look up it should show where it installed it to)
   8. `ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value` should show `mca:mpi:base:param:mpi_built_with_cuda_support:value:true` if you're MPI has cuda support
5. Install anaconda
   1. Download from the anaconda website
   2. bash ./Anaconda3-2020.11-Linux-x86_64.sh
6. Install numba
   1. Make sure CUDA_HOME is to the path that specifies the cuda that you build OpenMPI with (link)[https://numba.pydata.org/numba-doc/latest/cuda/overview.html#cudatoolkit-lookup] export CUDA_HOME= /usr/local/cuda-11.2
   2. conda install numba
   3. conda install cudatoolkit
7. Run with cuda-aware MPI, you can send NDDeviceArrays over MPI! Device memory can be sent via MPI

## Sending a NDDeviceArray (array on the GPU) over MPI
```python
from mpi4py import MPI
from numba import cuda
import numpy as np

@cuda.jit()
def kernel(array_on_gpu):
    array_on_gpu[0] = 0.5 # FAST!

def main():
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        input_array = np.zeros((100,), dtype=np.float64)
        gpu_input_array = cuda.to_device(input_array)
        MPI.COMM_WORLD.send(gpu_input_array.get_ipc_handle(), dest=1)
    else:
        handle = MPI.COMM_WORLD.recv(source=0)
        received_gpu_input_array = handle.open() # FAST
        # received_gpu_input_array.copy_to_host() # SLOW
        kernel[32, 32](received_gpu_input_array)
        # handle.close() # SLOW

if __name__ == "__main__":
    main()
```
