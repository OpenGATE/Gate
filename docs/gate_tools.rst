.. toctree::
   :maxdepth: 1
   :numbered:

   
The `GateTools <https://github.com/OpenGATE/GateTools>`_ repository contains a list of python command line tools to facilitate Gate simulations running and analysis. 

To install the package, use ::

  pip install gatetools

Command line tools have a built-in help, type the name of the tools with the ''--help'' (or '-h') option. For example::

  gate_info -h

API: all commands are also available as python functions::

  import gatetools as gt
  gt.print_gate_info()
  gt.image_convert(inputImage, pixeltype)


Current list of tools::

  gate_info
  gate_image_convert
  gate_image_arithm
  gate_image_uncertainty
  gate_gamma_index
  
  
  
