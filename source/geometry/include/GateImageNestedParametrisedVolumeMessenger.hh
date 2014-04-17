/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateImageNestedParametrisedVolumeMessenger :
  \brief  Messenger of GateImageNestedParametrisedVolume.
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEIMAGENESTEDPARAMETRISEDVOLUMEMESSENGER__HH
#define GATEIMAGENESTEDPARAMETRISEDVOLUMEMESSENGER__HH

#include "GateVImageVolumeMessenger.hh"
#include "globals.hh"

class GateImageNestedParametrisedVolume;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateImageNestedParametrisedVolume
class GateImageNestedParametrisedVolumeMessenger : public GateVImageVolumeMessenger
{
public:
  GateImageNestedParametrisedVolumeMessenger(GateImageNestedParametrisedVolume* volume);
  ~GateImageNestedParametrisedVolumeMessenger();

  void SetNewValue(G4UIcommand*, G4String);
};
//-----------------------------------------------------------------------------

#endif
