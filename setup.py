import setuptools

# TODO: complete this
INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'Pillow', 'scikit-image', 'opencv-python', 'tqdm', 'imageio']

setuptools.setup(
    name='monoport',
    url='',
    description='', 
    version='0.0.2',
    author='Ruilong Li',
    author_email='ruilongl@usc.edu',    
    license='MIT License',
    packages=['monoport'],
    install_requires=INSTALL_REQUIREMENTS + [
        'human_inst_seg@git+https://github.com/Project-Splinter/human_inst_seg',
        'ImplicitSegCUDA@git+https://github.com/Project-Splinter/ImplicitSegCUDA',
        'streamer_pytorch@git+https://github.com/Project-Splinter/streamer_pytorch',
    ]
)
