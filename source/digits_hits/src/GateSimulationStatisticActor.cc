/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateSimulationStatisticActor :
  \brief
*/

#ifndef GATESIMULATIONSTATISTICACTOR_CC
#define GATESIMULATIONSTATISTICACTOR_CC

#include "GateSimulationStatisticActor.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "G4Event.hh"

double get_elapsed_time(const timeval &start, const timeval &end) {
  double elapsed = 0;
  elapsed += end.tv_sec + 1e-6*end.tv_usec;
  elapsed -= start.tv_sec + 1e-6*start.tv_usec;
  return elapsed;
}

std::string get_date_string() {
  time_t now = time(NULL);
  return std::string(ctime(&now));
}


//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateSimulationStatisticActor::GateSimulationStatisticActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateSimulationStatisticActor() -- begin"<<G4endl);
  //SetTypeName("SimulationStatisticActor");
  pActor = new GateActorMessenger(this);
  ResetData();
  GateDebugMessageDec("Actor",4,"GateSimulationStatisticActor() -- end"<<G4endl);
  gettimeofday(&start,NULL);
  startDateStr = get_date_string();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateSimulationStatisticActor::~GateSimulationStatisticActor()
{
  delete pActor;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateSimulationStatisticActor::Construct()
{
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback Begin of Run
void GateSimulationStatisticActor::BeginOfRunAction(const G4Run*r)
{
  if (mNumberOfRuns == 0) { gettimeofday(&start_afterinit,NULL); }
  GateVActor::BeginOfRunAction(r);
  mNumberOfRuns++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback Begin Event
void GateSimulationStatisticActor::BeginOfEventAction(const G4Event*e)
{
  if (e->GetNumberOfPrimaryVertex() > 0) {
    GateVActor::BeginOfEventAction(e);
    mNumberOfEvents++;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Track
void GateSimulationStatisticActor::PreUserTrackingAction(const GateVVolume * v, const G4Track*t)
{
  GateVActor::PreUserTrackingAction(v, t);
  mNumberOfTrack++;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateSimulationStatisticActor::UserSteppingAction(const GateVVolume * v, const G4Step * step)
{
  GateVActor::UserSteppingAction(v, step);
  mNumberOfSteps++;

  // Get if boundary is reach
  G4StepPoint* pPostStepP = step->GetPostStepPoint();
  if (pPostStepP->GetStepStatus() == fGeomBoundary) {
    mNumberOfGeometricalSteps++;
  }
  else {
    mNumberOfPhysicalSteps++;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateSimulationStatisticActor::SaveData()
{
  GateVActor::SaveData();
  timeval end;
  gettimeofday(&end,NULL);
  std::ofstream os;
  OpenFileOutput(mSaveFilename, os);

  double currentSimulationTime = GateApplicationMgr::GetInstance()->GetCurrentTime();

  double virtualStartTime = GateApplicationMgr::GetInstance()->GetVirtualTimeStart();
  double virtualStopTime  = GateApplicationMgr::GetInstance()->GetVirtualTimeStop();
  double startTime = GateApplicationMgr::GetInstance()->GetTimeStart();
  double stopTime  = GateApplicationMgr::GetInstance()->GetTimeStop();

  double elapsedSimulationTime = currentSimulationTime - startTime;
  if (virtualStartTime != -1) elapsedSimulationTime = currentSimulationTime - virtualStartTime;

  os << "# NumberOfRun    = " << mNumberOfRuns << std::endl
     << "# NumberOfEvents = " << mNumberOfEvents << std::endl
     << "# NumberOfTracks = " << mNumberOfTrack << std::endl
     << "# NumberOfSteps  = " << mNumberOfSteps << std::endl
     << "# NumberOfGeometricalSteps  = " << mNumberOfGeometricalSteps << std::endl
     << "# NumberOfPhysicalSteps     = " << mNumberOfPhysicalSteps << std::endl
     << "# ElapsedTime           = " << get_elapsed_time(start,end) << std::endl
     << "# ElapsedTimeWoInit     = " << get_elapsed_time(start_afterinit,end) << std::endl
     << "# StartDate             = " << startDateStr
     << "# EndDate               = " << get_date_string()
     << "# StartSimulationTime        = " << startTime/s << std::endl
     << "# StopSimulationTime         = " << stopTime/s << std::endl
     << "# CurrentSimulationTime      = " << currentSimulationTime/s << std::endl
     << "# VirtualStartSimulationTime = " << virtualStartTime/s << std::endl
     << "# VirtualStopSimulationTime  = " << virtualStopTime/s << std::endl
     << "# ElapsedSimulationTime      = " << elapsedSimulationTime/s << std::endl;
  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mSaveFilename << G4endl);
  }
  os.flush();
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSimulationStatisticActor::ResetData()
{
  mNumberOfRuns = 0;
  mNumberOfEvents = 0;
  mNumberOfTrack = 0;
  mNumberOfGeometricalSteps = 0;
  mNumberOfPhysicalSteps = 0;
  mNumberOfSteps = 0;
  gettimeofday(&start,NULL);
  startDateStr = get_date_string();
}
//-----------------------------------------------------------------------------


#endif /* end #define GATESIMULATIONSTATISTICACTOR_CC */
