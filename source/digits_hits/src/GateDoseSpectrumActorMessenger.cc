/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateDoseSpectrumActorMessenger.hh"

//#ifdef G4ANALYSIS_USE_ROOT

#include "GateDoseSpectrumActor.hh"

//-----------------------------------------------------------------------------
GateDoseSpectrumActorMessenger::GateDoseSpectrumActorMessenger(GateDoseSpectrumActor * v)
: GateActorMessenger(v),
  pDoseSpectrumActor(v)
{
  BuildCommands(baseName+v->GetObjectName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateDoseSpectrumActorMessenger::~GateDoseSpectrumActorMessenger()
{
  if(pDosePrimaryOnlyCmd) delete pDosePrimaryOnlyCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/dosePrimaryOnly";
  pDosePrimaryOnlyCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String( "calculate dose primary only");
  pDosePrimaryOnlyCmd->SetGuidance( guidance);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if( cmd == pDosePrimaryOnlyCmd)
      pDoseSpectrumActor->DosePrimaryOnly(pDosePrimaryOnlyCmd->GetNewBoolValue(newValue));

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

//#endif
