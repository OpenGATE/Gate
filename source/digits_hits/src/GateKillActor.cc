/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class GateKillActor :
  \brief
*/

#include "GateKillActor.hh"

#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateKillActor::GateKillActor(G4String name, G4int depth):GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateKillActor() -- begin\n");
  pMessenger = new GateActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateKillActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateKillActor::~GateKillActor()
{
  GateDebugMessageInc("Actor",4,"~GateKillActor() -- begin\n");
  delete pMessenger;
  GateDebugMessageDec("Actor",4,"~GateKillActor() -- end\n");
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
  GateVActor::SaveData();
  if (mSaveFilename == "FilenameNotGivenForThisActor") return;
  std::ofstream os;
  OpenFileOutput(mSaveFilename, os);
  os << "# NumberOfKillTracks = " << mNumberOfTrack << Gateendl;
  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mSaveFilename << Gateendl);
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
