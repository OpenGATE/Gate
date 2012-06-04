/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \brief Class GateStopOnScriptActor : 
  \brief 
 */

#ifndef GATESTOPONSCRIPTACTOR_CC
#define GATESTOPONSCRIPTACTOR_CC

#include "GateStopOnScriptActor.hh"
#include "GateUserActions.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateStopOnScriptActor::GateStopOnScriptActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateStopOnScriptActor() -- begin"<<G4endl);  
  pMessenger = new GateStopOnScriptActorMessenger(this);
  mSaveAllActors = false;
  GateDebugMessageDec("Actor",4,"GateStopOnScriptActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateStopOnScriptActor::~GateStopOnScriptActor() 
{
  GateDebugMessageInc("Actor",4,"~GateStopOnScriptActor() -- begin"<<G4endl);
  GateDebugMessageDec("Actor",4,"~GateStopOnScriptActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateStopOnScriptActor::Construct()
{
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableEndOfEventAction(true);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateStopOnScriptActor::SaveData()
{
  std::string cmd("source ");
  cmd += mSaveFilename;
  cmd += " "+DoubletoString(GateUserActions::GetUserActions()->GetCurrentEventNumber());
  if (mSaveAllActors) { // enable save of all actors (except me !!)
    GateMessage("Actor", 0, "GateStopOnScriptActor -- saving actors" << G4endl);
    std::vector<GateVActor*> &l = GateActorManager::GetInstance()->GetTheListOfActors();
    std::vector<GateVActor*>::iterator sit;
    for(sit = l.begin(); sit!=l.end(); ++sit) {
      GateMessage("Actor", 1, "GateStopOnScriptActor -- save actor " << (*sit)->GetName() << G4endl);
      if (*sit != this) (*sit)->SaveData();
    }
  }

  GateMessage("Actor", 0, "GateStopOnScriptActor -- executing command '" << cmd << "'" << G4endl);
  int r = system(cmd.c_str());
  GateMessage("Actor", 0, "GateStopOnScriptActor -- executed command returncode=" << r << G4endl);

  if (GateActorManager::GetInstance()->GetResetAfterSaving()) {
    GateMessage("Actor", 0, "GateStopOnScriptActor -- resetting actors" << G4endl);
    std::vector<GateVActor*> &l = GateActorManager::GetInstance()->GetTheListOfActors();
    std::vector<GateVActor*>::iterator sit;
    for(sit = l.begin(); sit!=l.end(); ++sit) {
      GateMessage("Actor", 1, "GateStopOnScriptActor -- reseting actor " << (*sit)->GetName() << G4endl);
      if (*sit != this) (*sit)->ResetData();
    }
  }
  if (r == 0) return;
  else exit(0);
}

void GateStopOnScriptActor::ResetData() {}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateStopOnScriptActor::EnableSaveAllActors(bool b) 
{
  mSaveAllActors = b ;
}
//-----------------------------------------------------------------------------

#endif /* end #define GATESTOPONSCRIPTACTOR_CC */

