/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateImageRegionalizedSubVolumeMessenger :
  \brief  Messenger of GateImageRegionalizedSubVolume.
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateImageRegionalizedSubVolumeMessenger__hh__
#define __GateImageRegionalizedSubVolumeMessenger__hh__

#include "GateVolumeMessenger.hh"
#include "globals.hh"

class GateImageRegionalizedSubVolume;

//====================================================================
/// \brief Messenger of GateImageRegionalizedSubVolume
class GateImageRegionalizedSubVolumeMessenger : public GateVolumeMessenger
{
public:
  GateImageRegionalizedSubVolumeMessenger(GateImageRegionalizedSubVolume* volume);
  ~GateImageRegionalizedSubVolumeMessenger();

  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateImageRegionalizedSubVolume* pVolume;

};
//====================================================================

#endif
