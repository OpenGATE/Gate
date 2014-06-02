/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
   \class GateImageRegionalizedVolumeMessenger
   \author thibault.frisson@creatis.insa-lyon.fr
           laurent.guigues@creatis.insa-lyon.fr
	   david.sarrut@creatis.insa-lyon.fr
*/

 #ifndef __GateImageRegionalizedVolumeMessenger__hh__
#define __GateImageRegionalizedVolumeMessenger__hh__

#include "GateVImageVolumeMessenger.hh"
#include "globals.hh"
#include "G4UIcmdWithAString.hh"

class GateImageRegionalizedVolume;

//====================================================================
/// \brief Messenger of GateImageRegionalizedVolume
class GateImageRegionalizedVolumeMessenger : public GateVImageVolumeMessenger
{
public:
  GateImageRegionalizedVolumeMessenger(GateImageRegionalizedVolume* volume);
  ~GateImageRegionalizedVolumeMessenger();
    
  void SetNewValue(G4UIcommand*, G4String);

private:
  GateImageRegionalizedVolume* pVolume;   
  G4UIcmdWithAString* pDistanceMapNameCmd;
};
//====================================================================

#endif

