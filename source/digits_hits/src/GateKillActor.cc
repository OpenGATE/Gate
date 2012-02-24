/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateKillActor : 
  \brief 
*/

#ifndef GATEKILLACTOR_CC
#define GATEKILLACTOR_CC

#include "GateKillActor.hh"

#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateKillActor::GateKillActor(G4String name, G4int depth):GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateKillActor() -- begin"<<G4endl);
  //SetTypeName("KillActor");
  pActor = new GateActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateKillActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateKillActor::~GateKillActor() 
{
  GateDebugMessageInc("Actor",4,"~GateKillActor() -- begin"<<G4endl);
  GateDebugMessageDec("Actor",4,"~GateKillActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateKillActor::Construct()
{
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4bool GateKillActor::ProcessHits(G4Step * step , G4TouchableHistory* )
{
  step->GetTrack()->SetTrackStatus( fStopAndKill );
  mNumberOfTrack++;
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateKillActor::SaveData()
{
  if (mSaveFilename == "FilnameNotGivenForThisActor") return;
  std::ofstream os;
  OpenFileOutput(mSaveFilename, os);
  os << "# NumberOfKillTracks = " << mNumberOfTrack << std::endl;
  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mSaveFilename << G4endl);
  }
  os.flush();
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateKillActor::ResetData() 
{
  mNumberOfTrack = 0;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATEKILLACTOR_CC */

