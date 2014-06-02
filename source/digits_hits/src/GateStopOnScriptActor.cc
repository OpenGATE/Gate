/*----------------------
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
#include <signal.h>
#include <sys/wait.h>

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
  GateVActor::SaveData();
  std::string cmd("bash ");
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

  // http://stackoverflow.com/questions/11084959/system-always-returns-non-zero-when-called-as-cgi
  signal(SIGCHLD,SIG_DFL);

  // Count the number of time we try to "system"
  static int n = 0;
  static int max_n = 10;

  /*
  int a = system("which data/bidon.sh");
  DD(a);
  DD(WIFEXITED(a));
  DD(WEXITSTATUS(a));

  int b = system("TOTOwhich data/bidon.sh");
  DD(b);
  DD(WIFEXITED(b));
  DD(WEXITSTATUS(b));

  int c = system("source data/bidon.sh");
  DD(c);
  DD(WIFEXITED(c));
  DD(WEXITSTATUS(c));

  int d = system("./data/bidon.sh");
  DD(d);
  DD(WIFEXITED(d));
  DD(WEXITSTATUS(d));
  */

  {
	  std::string s("ls ");
	  s += mSaveFilename;
	  int e = system(s.c_str());
	  DD(e);
	  DD(WIFEXITED(e));
	  DD(WEXITSTATUS(e));

	  if (WIFEXITED(e) && WEXITSTATUS(e) == 2) {
		  n++;
		  GateMessage("Actor", 0, "GateStopOnScriptActor -- file '" << mSaveFilename << "' not found. try to continue"
				  << n << " / " << max_n << G4endl);
		  return;
	  }
  }


  int r = system(cmd.c_str());
  GateMessage("Actor", 0, "GateStopOnScriptActor -- executed command returncode=" << r << G4endl);
  GateMessage("Actor", 0, "GateStopOnScriptActor -- executed command WIFEXITED=" << WIFEXITED(r) << G4endl);
  GateMessage("Actor", 0, "GateStopOnScriptActor -- executed command WEXITSTATUS=" << WEXITSTATUS(r) << G4endl);

  if (!WIFEXITED(r)) {
    /* the program didn't terminate normally */
    n++;
    GateMessage("Actor", 0, "GateStopOnScriptActor -- Could not 'system'. I try to continue "
                << n << " / " << max_n << G4endl);
  }
  else {
    if (WEXITSTATUS(r) == 127) {
      /* command failed */
      n++;
      GateMessage("Actor", 0, "GateStopOnScriptActor -- command fail. I try to continue "
                  << n << " / " << max_n << G4endl);
    }
  }

  if (GateActorManager::GetInstance()->GetResetAfterSaving()) {
    GateMessage("Actor", 0, "GateStopOnScriptActor -- resetting actors" << G4endl);
    std::vector<GateVActor*> &l = GateActorManager::GetInstance()->GetTheListOfActors();
    std::vector<GateVActor*>::iterator sit;
    for(sit = l.begin(); sit!=l.end(); ++sit) {
      GateMessage("Actor", 1, "GateStopOnScriptActor -- reseting actor " << (*sit)->GetName() << G4endl);
      if (*sit != this) (*sit)->ResetData();
    }
  }

  if (n >= max_n) {
    GateMessage("Actor", 0, "GateStopOnScriptActor -- too much trial. I stop. " << n << " / " << max_n << G4endl);
    exit(0);
  }

  //if (r == 0) return;
  if (WEXITSTATUS(r) == 1) {
    GateMessage("Actor", 0, "GateStopOnScriptActor -- return is 1 ; completed. I stop now." << G4endl);
    exit(0);
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateStopOnScriptActor::ResetData() {}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateStopOnScriptActor::EnableSaveAllActors(bool b)
{
  mSaveAllActors = b ;
}
//-----------------------------------------------------------------------------

#endif /* end #define GATESTOPONSCRIPTACTOR_CC */
