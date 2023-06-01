/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \brief Class GateMaterialFilterMessenger
*/

#ifndef GATEMATFILTERMESSENGER_CC
#define GATEMATFILTERMESSENGER_CC

#include "GateMaterialFilterMessenger.hh"

#include "GateMaterialFilter.hh"


//-----------------------------------------------------------------------------
GateMaterialFilterMessenger::GateMaterialFilterMessenger(GateMaterialFilter* matFilter)

  : pMaterialFilter(matFilter)
{
  BuildCommands(pMaterialFilter->GetObjectName());

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialFilterMessenger::~GateMaterialFilterMessenger()
{
  delete pAddMaterialCmd;
  delete pInvertCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
  
  bb = base+"/addMaterial";
  pAddMaterialCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add material";
  pAddMaterialCmd->SetGuidance(guidance);
  pAddMaterialCmd->SetParameterName("Material name",false);

  bb = base+"/invert";
  pInvertCmd = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Invert the filter";
  pInvertCmd->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddMaterialCmd)
      pMaterialFilter->Add(param);
  if(command==pInvertCmd)
      pMaterialFilter->setInvert();
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEMATERIALFILTERMESSENGER_CC */
