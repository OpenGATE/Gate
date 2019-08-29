
GateTools
!!!!!!!!!!!!!!!

.. toctree::
   :maxdepth: 1
   :numbered:

   
The `GateTools <https://github.com/OpenGATE/GateTools>`_ repository contains a list of python command line tools to facilitate Gate simulations running and analysis. 

Install with : `pip install gatetools`

Example of usage: 

.. code-block::
   gate_image_convert -i input.dcm -o output.mhd  
   gate_image_convert -i input.mhd -o output_float.mhd -p float  
   gate_image_arithm -i *.mhd -o output.mhd -O sum  
   gate_gamma_index data/tps_dose.mhd result.XYZ/gate-DoseToWater.mhd -o gamma.mhd --dd 2 --dta 2.5 -u "%" -T 0.2

Current list of command line tools. Use the flag `-h` to get print the help of each tool.

+-------------------------+------------+
| Tool                    | Comment    |
+ ------------------------+ ----------- +
| `gate_info`             | Display info about current Gate/G4 version  |
| `gate_image_arithm`     | Pixel- or voxel-wise arithmetic operations |
| `gate_image_convert`    | Convert image file format (dicom, mhd, hdr, nfty ... ) |
| `gate_image_uncertainty`| Compute statistical uncertainty|
| `gate_gamma_index`      | Compute gamma index between images|

All tools are also available to be use within your own Python script with, for example: 

.. code-block:: python
   import gatetools as gt  
   gt.image_convert(inputImage, pixeltype)

Tests: run 

  python -m unittest gatetools

When developing, install with: 

  pip install -e . 

Dependencies:
- click: https://click.palletsprojects.com
- uproot: https://github.com/scikit-hep/uproot
- pydicom: https://pydicom.github.io/
- tqdm: https://github.com/tqdm/tqdm
- colored: https://gitlab.com/dslackw/colored
- itk: https://itk.org/
