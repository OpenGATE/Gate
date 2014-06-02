/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateImageRegularParametrisedVolumeMessenger :
  \brief  Messenger of GateImageRegularParametrisedVolume.
*/

 #ifndef __GateImageRegularParametrisedVolumeMessenger__hh__
#define __GateImageRegularParametrisedVolumeMessenger__hh__

#include "GateVImageVolumeMessenger.hh"
#include "globals.hh"
#include "G4UIcmdWithABool.hh"

class GateImageRegularParametrisedVolume;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateImageRegularParametrisedVolume
class GateImageRegularParametrisedVolumeMessenger : public GateVImageVolumeMessenger
{
public:
  GateImageRegularParametrisedVolumeMessenger(GateImageRegularParametrisedVolume* volume);
  ~GateImageRegularParametrisedVolumeMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  GateImageRegularParametrisedVolume* pVolume;
  G4UIcmdWithABool* SkipEqualMaterialsCmd;
};
//-----------------------------------------------------------------------------

#endif
