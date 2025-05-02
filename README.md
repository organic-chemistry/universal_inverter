mamba env create -f environment-gpu.yml


mamba env create -f environment-cuda-11.4.yml
pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install optax jax==0.4.2 jaxlib==0.4.2+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install matplotlib
pip install flax
