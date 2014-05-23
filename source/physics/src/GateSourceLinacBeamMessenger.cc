/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceLinacBeamMessenger.hh"
#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//----------------------------------------------------------------------------------------
GateSourceLinacBeamMessenger::GateSourceLinacBeamMessenger(GateSourceLinacBeam * source)
  :GateVSourceMessenger(source), mSource(source)
{ 
  G4String cmdName;
  cmdName = GetDirectoryName()+"setReferencePosition";
  mRefPosCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  mRefPosCmd->SetGuidance("Set phase space reference position (translate coordinate in phase space with this position).");

  cmdName = GetDirectoryName()+"setSourceFromPhaseSpaceFilename";
  mSourceFromPhSCmd = new G4UIcmdWithAString(cmdName,this);
  mSourceFromPhSCmd->SetGuidance("Set source (root file) filename (computed from phase space).");

  cmdName = GetDirectoryName()+"setMaxLeafDistanceAccordingToTime";
  mRmaxCmd = new G4UIcmdWithAString(cmdName,this);
  mRmaxCmd->SetGuidance("Set filename with max radius according to time.");
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourceLinacBeamMessenger::~GateSourceLinacBeamMessenger() {
  delete mRefPosCmd;
  delete mSourceFromPhSCmd;
  delete mRmaxCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceLinacBeamMessenger::SetNewValue(G4UIcommand* command,G4String newValue) { 
  if( command == mRefPosCmd ) {
   mSource->SetReferencePosition(mRefPosCmd->GetNew3VectorValue(newValue));
  }
  if( command == mSourceFromPhSCmd ) {
   mSource->SetSourceFromPhaseSpaceFilename(newValue);
  }
  if( command == mRmaxCmd ) {
   mSource->SetRmaxFilename(newValue);
  }
  GateVSourceMessenger::SetNewValue(command,newValue);
}
//----------------------------------------------------------------------------------------


