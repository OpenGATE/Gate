/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

//LF
//#include <strstream>
#include <sstream>
//LF
#include "GateUIterminal.hh"


G4int GateUIterminal::ReceiveG4cout(const G4String& coutString)
{
  if ( coutString == "G4PhysicalVolumeModel::Validate() called.\n" )
    return 0;
  if ( coutString == "  Volume of the same name and copy number (\"world_P\", copy 0) still exists and is being used.\n" )
    return 0;
  if ( coutString == "  Be warned that this does not necessarily guarantee it's the same\n" )
    return 0;
  if ( coutString == "  volume you originally specified in /vis/scene/add/.\n" )
    return 0;

  std::cout << coutString << std::flush;
  return 0;
}


G4int GateUIterminal::ReceiveG4cerr(const G4String& cerrString)
{
	std::cerr << "[G4-cerr] " <<  cerrString << std::flush;
	// Check if this error is 'command not found' (or related) to stop Gate
    bool isMacroError = false;
    std::string::size_type i = cerrString.find("***** COMMAND NOT FOUND <", 0);
    isMacroError = isMacroError || (i != std::string::npos);
    i = cerrString.find("***** Illegal application state <", 0);
    isMacroError = isMacroError || (i != std::string::npos);
    i = cerrString.find("***** Illegal parameter (", 0);
    isMacroError = isMacroError || (i != std::string::npos);
    i = cerrString.find("***** Can not open a macro file <", 0);
    isMacroError = isMacroError || (i != std::string::npos);
    i = cerrString.find("ERROR: Can not open a macro file <", 0);
    isMacroError = isMacroError || (i != std::string::npos);

    if (isMacroError) {
      std::cerr << "[Gate] Sorry, error in a macro command : abort.\n";
      if (G4StrUtil::contains(cerrString, "digitizer"))
            {
          	  std::cerr << "------------------------ [GateSinglesDigitizer Messenger Problem]------------------------ \n";
          	  std::cerr << "Probably you try to use a command for the old version of the digitizer.\n";
          	  std::cerr << "Try </gate/digitizerMgr> commands instead, you can find the documentation here: XXXX \n"; //TODO insert the link to the documentation page
          	  std::cerr << "A tool, gt_digi_mac_converter, to automatically convert your old digitizer macro to a new is also available here: https://github.com/OpenGATE/GateTools \n\n ";
          	  std::cerr << "To use gt_digi_mac_converter tool, please, do: \n";
          	  std::cerr << "pip install gatetools\n";
          	  std::cerr << "git clone --recursive https://github.com/OpenGATE/GateTools.git\n";
          	  std::cerr << "cd GateTools\n";
          	  std::cerr << "pip install -e .\n";
          	  std::cerr << "export PATH=\"<YOUR PATH>/GateTools/bin:$PATH\" \n\n";
          	  std::cerr << "gt_digi_mac_converter -i digitizer_old.mac -o digitizer_new.mac -sd <SDname> -multi SinglesDigitizer \n";
          	  std::cerr << "where -i defines input old digitizer macro,\n -o defines output new digitizer macro,\n -sd defines the sensitive detector name (the same as in /gate/<SDname>/attachCrystalSD),\n -multi <mode> is the option if you have several SinglesDigitizers or CoincidenceSorters, where <mode> = SinglesDigitizer or CoincidenceSorter\n";
          	  std::cerr << "---------------------------------------------------------------------------------- \n";
            }

      exit(-1);
    }

  return 0;
}

