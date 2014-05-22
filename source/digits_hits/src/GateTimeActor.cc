/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATETIMEACTOR_CC
#define GATETIMEACTOR_CC

#include "GateTimeActor.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "GateTimeActorMessenger.hh"

#include "G4Event.hh"
#include "G4VProcess.hh"
#include "G4SteppingManager.hh"
#include "G4EventManager.hh"

//-----------------------------------------------------------------------------
/// Constructors
GateTimeActor::GateTimeActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateTimeActor() -- begin"<<G4endl);
  ResetData();
  mDetailedStatFlag = false;
  pMessenger = new GateTimeActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateTimeActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTimeActor::~GateTimeActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::EnableDetailedStats(bool b)
{
  mDetailedStatFlag = b;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateTimeActor::Construct()
{
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  ResetData();
  mNumberOfEvents = 0;
  mTotalEventUserTime = 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::BeginOfRunAction(const G4Run*r)
{
  GateVActor::BeginOfRunAction(r);
  mCurrentRunTimer.Start();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::EndOfRunAction(const G4Run*r)
{
  GateVActor::EndOfRunAction(r);
  mCurrentRunTimer.Stop();

  UpdateCurrentTextOutput();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::UpdateCurrentTextOutput()
{
  std::ostringstream ss;
  ss << mCurrentRunTimer << std::endl;
  ss << "Mean Event time = " << mTotalEventUserTime/mNumberOfEvents
            << " " << mNumberOfEvents << " " << mTotalEventUserTime << std::endl;
  ss << "Mean Track time = " << mTotalTrackUserTime/mNumberOfTracks
            << " " << mNumberOfTracks << " " << mTotalTrackUserTime << std::endl;
  ss << "Mean Step time = " << mTotalStepUserTime/mNumberOfSteps
            << " " << mNumberOfSteps << " " << mTotalStepUserTime << std::endl;
  ss << "PPS = " << mNumberOfEvents/mTotalEventUserTime << std::endl;
  ss << "SPS = " << mNumberOfSteps/mTotalEventUserTime << std::endl;

  ss << std::endl << "Time per particle " << std::endl;
  MapType::iterator iter;
  MapType::iterator iterT = mTrackPerParticle.begin();
  for(iter = mTimePerParticle.begin(); iter != mTimePerParticle.end(); ++iter) {
    ss << iter->first << " " << iter->second << " " << iterT->second << std::endl;
    ++iterT;
  }

  ss << std::endl << "Limiting process" << std::endl;
  for(iter = mNumberOfLimitingProcess.begin(); iter != mNumberOfLimitingProcess.end(); ++iter) {
    ss << iter->first << " " << iter->second << " " << std::endl;
  }

  ss << std::endl << "Along process" << std::endl;
  for(iter = mNumberOfAlongByProcess.begin(); iter != mNumberOfAlongByProcess.end(); ++iter) {
    ss << iter->first << " " << iter->second << " " << std::endl;
  }

  mCurrentTextOutput = ss.str();
  // std::cout << mCurrentTextOutput;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::BeginOfEventAction(const G4Event*e)
{
  // Do nothing if no vertex
  if (e->GetNumberOfPrimaryVertex() <= 0) return;
  GateVActor::BeginOfEventAction(e);
  // DD("Start current Event timer");
  mCurrentEventTimer.Start();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::EndOfEventAction(const G4Event*e)
{
  GateVActor::EndOfEventAction(e);
  mCurrentEventTimer.Stop();
  // DD("Stop current Event timer");
  // std::cout << mCurrentEventTimer << std::endl;
  mTotalEventUserTime += mCurrentEventTimer.GetUserElapsed();
  // DD(mTotalEventUserTime);
  mNumberOfEvents++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::PreUserTrackingAction(const GateVVolume * v, const G4Track*t)
{
  GateVActor::PreUserTrackingAction(v, t);
  // DD("Start current Track timer");
  if (mDetailedStatFlag) {
    mCurrentParticleName = t->GetParticleDefinition()->GetParticleName();
  }
  mCurrentTrackTimer.Start();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::PostUserTrackingAction(const GateVVolume * v, const G4Track * track)
{
  GateVActor::PostUserTrackingAction(v, track);
  mCurrentTrackTimer.Stop();

  // DD("Stop current Track timer");
  // std::cout << mCurrentTrackTimer << std::endl;

  // Total time
  double t = mCurrentTrackTimer.GetUserElapsed();
  mTotalTrackUserTime += t;
  // DD(mTotalTrackUserTime);

  if (mDetailedStatFlag) {
    // Time per particle
    mTimePerParticle[mCurrentParticleName] += t;
    mTrackPerParticle[mCurrentParticleName]++;
  }

  mNumberOfTracks++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::UserSteppingAction(const GateVVolume * v, const G4Step * step)
{
  GateVActor::UserSteppingAction(v, step);
  if (mNumberOfSteps != 0) { // the first step does not count
    mCurrentStepTimer.Stop();

    double t = mCurrentStepTimer.GetUserElapsed();

    // Total time
    mTotalStepUserTime += t;

    if (mDetailedStatFlag) {
      // Count according to processes
      // http://geant4.slac.stanford.edu/Tips/event/6.html

      // Limiting step
      G4String limitingProcess = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
      mNumberOfLimitingProcess[limitingProcess]++;

      // Along per process
      G4SteppingManager* fpSteppingManager =
        G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();
      G4StepStatus stepStatus = fpSteppingManager->GetfStepStatus();
      if(stepStatus!=fExclusivelyForcedProc && stepStatus!=fAtRestDoItProc) {
        G4ProcessVector* procAlong = fpSteppingManager->GetfAlongStepDoItVector();
        size_t MAXofAlongStepLoops = fpSteppingManager->GetMAXofAlongStepLoops();
        for(size_t i2=0;i2<MAXofAlongStepLoops;i2++) {
          if((*procAlong)[i2]!=0) {
            G4String proc = (*procAlong)[i2]->GetProcessName();
            G4String n = proc+"-"+mCurrentParticleName;
            mNumberOfAlongByProcess[n]++;
          }
        }
      }
    } // end detailed stats

  } // end mNumberOfSteps != 0

  // Start timer for next step
  mNumberOfSteps++;
  mCurrentStepTimer.Start();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateTimeActor::SaveData()
{
  GateVActor::SaveData();
  UpdateCurrentTextOutput();
  std::ofstream os;
  OpenFileOutput(mSaveFilename, os);
  os << mCurrentTextOutput;
  os.flush();
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActor::ResetData()
{
  mNumberOfEvents = 0;
  mNumberOfTracks = 0;
  mNumberOfSteps = 0;
  mTotalStepUserTime = 0;
  mTotalTrackUserTime = 0;
  mTotalEventUserTime = 0;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATETIMEACTOR_CC */
