/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class GateVImageVolumeMessenger : 
  \brief Messenger of GateVImageVolume.
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */


#ifndef __GateVImageVolumeMessenger__hh__
#define __GateVImageVolumeMessenger__hh__

#include "GateVolumeMessenger.hh"
#include "globals.hh"

class GateVImageVolume;

class G4UIcmdWithAString;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithABool;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageVolume
class GateVImageVolumeMessenger : public GateVolumeMessenger
{
public:
  GateVImageVolumeMessenger(GateVImageVolume* volume);
  ~GateVImageVolumeMessenger();
    
  void SetNewValue(G4UIcommand* cmd=0, G4String = " ");

private:
  GateVImageVolume * pVImageVolume; 

  G4UIcmdWithAString        * pImageFileNameCmd;
  G4UIcmdWithAString        * pImageFileNameCmdDeprecated;
  G4UIcmdWithAString        * pLabelToMaterialFileNameCmd;
  G4UIcmdWithAString        * pHUToMaterialFileNameCmd;
  G4UIcmdWithAString        * pHUToMaterialFileNameCmdDeprecated;
  G4UIcmdWithAString        * pRangeMaterialFileNameCmd;
  G4UIcmdWith3VectorAndUnit * pIsoCenterCmd;
  G4UIcmdWith3VectorAndUnit * pSetOriginCmd;
  G4UIcmdWithAString        * pBuildDistanceTransfoCmd;
  G4UIcmdWithAString        * pBuildLabeledImageCmd;
  G4UIcmdWithABool          * pDoNotBuildVoxelsCmd;
};
//-----------------------------------------------------------------------------

#endif

