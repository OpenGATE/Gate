/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateApplicationMgr.hh"
#include "GateApplicationMgrMessenger.hh"
#include "GateClock.hh"
#include "G4UImanager.hh"
#include "GateOutputMgr.hh"
#include "GateRunManager.hh"
#include "GateRandomEngine.hh"
#include "GateDetectorConstruction.hh"
#include "GateVVolume.hh"
#include "GateObjectStore.hh"
#include "GateMiscFunctions.hh"
#include "GateVSource.hh"
#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"

GateApplicationMgr* GateApplicationMgr::instance = 0; 
//------------------------------------------------------------------------------------------
GateApplicationMgr::GateApplicationMgr(): 
  nVerboseLevel(0), m_pauseFlag(false), m_exitFlag(false), 
  mOutputMode(true),  mTimeSliceIsSetUsingAddSlice(false), mTimeSliceIsSetUsingReadSliceInFile(false),
  mTimeStepInTotalAmountOfPrimariesMode(0.0)
{
  if(instance != 0)
    { G4Exception( "GateApplicationMgr::GateApplicationMgr", "GateApplicationMgr", FatalException, "GateApplicationMgr constructed twice."); }
  m_appMgrMessenger = new GateApplicationMgrMessenger();

  mTimeSliceDuration = 0;
  mTimeSlices.push_back(0);
  mTimeSlices.push_back(1*s);   // default to a single, 1s run with one time slice

  mRequestedAmountOfPrimaries = 0;
  mRequestedAmountOfPrimariesPerRun = 0;
  mATotalAmountOfPrimariesIsRequested = false;
  mAnAmountOfPrimariesPerRunIsRequested = false;

  m_weight=-1.;

  // We initialize virtual times to -1. to be able to later recognize if we are in cluster mode or not
  m_virtualStart = -1.;
  m_virtualStop = -1.;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
GateApplicationMgr::~GateApplicationMgr()
{
  delete m_appMgrMessenger;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTotalNumberOfPrimaries(double n) { 
  if(mATotalAmountOfPrimariesIsRequested) GateError("You have already defined a total number of primaries or a number of primaries per run");
  mRequestedAmountOfPrimaries = (long int)lrint(n); 
  mATotalAmountOfPrimariesIsRequested = true; 
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetNumberOfPrimariesPerRun(double n) { 
  if(mATotalAmountOfPrimariesIsRequested) GateError("You have already defined a total number of primaries or a number of primaries per run");
  mRequestedAmountOfPrimariesPerRun = (long int)lrint(n); 
  mATotalAmountOfPrimariesIsRequested = true; 
  mAnAmountOfPrimariesPerRunIsRequested = true;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetNoOutputMode() {
  mOutputMode = false;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
//void GateApplicationMgr::EnableSuccessiveSourceMode(bool t) {
//  mSuccessiveSourceMode = t;
//}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
//bool GateApplicationMgr::IsSuccessiveSourceModeIsEnabled() {
//  return mSuccessiveSourceMode;
//}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::ReadTimeSlicesInAFile(G4String filename) {
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateError("Please do not use 'readTimeSlicesIn' twice");
  }
  if (mTimeSliceDuration != 0.0) {
    GateError("Please do not use 'addSlice' or 'readTimeSlicesIn' commands with 'setTimeSlice' command");
  }
  if (mTimeSliceIsSetUsingAddSlice) {
    GateError("Please do not use 'addSlice' and 'readTimeSlicesIn' commands at the same time");
  }

  // TODO: this does nothing for now. Fix later. JS 28/10/2015
  // Open file  
  std::ifstream is;
  OpenFileInput(filename, is);
  skipComment(is);
  
  // Use Time
  double timeUnit=0.;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << filename << "' need to begin with 'Time'\n");
  }

  skipComment(is);
  double t = ReadDouble(is)*timeUnit; // read first time
  mTimeSlices.resize(1);
  mTimeSlices[0] = t;
  skipComment(is); // just in case the user felt like cluttering up the time slice list with comments...
  
  while (is)
  {
    t = ReadDouble(is)*timeUnit;
    if (t < mTimeSlices.back())
    {
      GateError("Time slices should be in increasing order, but I read " << t/s
                << " sec after " << mTimeSlices.back()/s << " sec.\n");
      exit(-1);
    }
    t -= mTimeSlices.back();
    mTimeSlices.push_back(t);

    skipComment(is);
  }

  is.close();

  mTimeSliceIsSetUsingReadSliceInFile = true;

}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeSlice(G4double timeSlice) 
{
  if (mTimeSliceIsSetUsingReadSliceInFile || mTimeSliceIsSetUsingAddSlice) {
    GateError("Please do not use 'setTimeSlice' command with 'addTimeSlice' or 'readTimeSlicesIn' commands");
  }
  mTimeSliceDuration = timeSlice;
  //if (nVerboseLevel>0) G4cout << "Time Slice set to (s) " << m_timeSlice/s << Gateendl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeInterval(G4double v)
{
  // this should really be called AddSlice or something similar to the macro command that invokes it
  if (mTimeSliceDuration!=0.0) {
    GateError("Please do not use 'addSlice' commands with 'setTimeSlice' command");
  }
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateError("Please do not use 'addSlice' and 'readTimeSlicesIn' commands at the same time");
  }

  if(mTimeSliceIsSetUsingAddSlice) // we've already added a slice previously
    mTimeSlices.push_back(mTimeSlices.back()+v);
  else
  {
    mTimeSlices[1] = mTimeSlices[0] + v;
    mTimeSliceIsSetUsingAddSlice = true;
  }

  // listOfTimeSlice.push_back(v);
  //if (nVerboseLevel>0) G4cout << "Time Slice set to (s) " << m_timeSlice/s << Gateendl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeSlice() 
{
  return mTimeSliceDuration;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeSlice(int run) 
{
  //if(listOfTimeSlice.size()==0) return m_timeSlice;

  if (run>(int)(mTimeSlices.size()-2)) {
    GateWarning("Warning in GateApplicationMgr::GetTimeSlice, run=" << run << " is greater than the list of slices. Do nothing\n");
    return 0.0; // DO NOTHING in this case
  }
  return (mTimeSlices[run+1]-mTimeSlices[run]);
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetEndTimeSlice(int run) 
{
  //if(listOfTimeSlice.size()==0) return m_timeSlice;

  if (run>(int)(mTimeSlices.size()-2)) {
    GateError("Error in GateApplicationMgr::GetEndTimeSlice, run=" << run << "\n");
  }

  return min(mTimeSlices[run+1],m_virtualStop);
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeStart(G4double timeStart) 
{
  if (timeStart<0.0) {
    GateError("setTimeStart should not be negative");
  }
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateError("setTimeStart command cannot be used with readTimeSlicesIn command. The first time in file is taken as timeStart.");
  }
  if (mTimeSliceIsSetUsingAddSlice) {
    GateWarning("The start time should be set before adding time slices.");
  }

  mTimeSlices[0] = timeStart;
  if (nVerboseLevel>0) G4cout << "Time Start set to (s) " << mTimeSlices[0]/s << Gateendl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeStart() 
{
  return mTimeSlices.front();
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeStop(G4double timeStop) 
{
  if (timeStop<0.0) {
    GateError("setTimeStop can not be negative");
  }
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateWarning("Stop time already set by readTimeSlicesIn command. Command ignored.");
    return;
  }
  if (mTimeSliceIsSetUsingAddSlice) {
    GateWarning("The stop time is calculated from the added time slices. Command ignored.");
    return;
  }

  mTimeSlices[1] = timeStop;
  if (nVerboseLevel>0) G4cout << "Time Stop set to (s) " << mTimeSlices.back()/s << Gateendl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeStop() 
{
  return mTimeSlices.back();
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetVirtualTimeStop() 
{
  return m_virtualStop;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetVirtualTimeStart() 
{
  return m_virtualStart;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StartDAQComplete(G4ThreeVector param) 
{
  SetTimeSlice(param[0]);
  SetTimeStart(param[1]);
  SetTimeStop(param[2]);
  StartDAQ();
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StartDAQ() 
{
  // With this method we check for all output module enabled but with no
  // filename given. In this case we disable the output module and send a warning.
  GateOutputMgr::GetInstance()->CheckFileNameForAllOutput();

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Source initialization =============\n");
 
  InitializeTimeSlices();

  // init sources if needed
  GateSourceMgr::GetInstance()->Initialization();

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Acquisition starts! =============\n");

  // Verbose 
  GateMessage("Acquisition", 0, "Simulation start time = " << mTimeSlices.front()/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation end time   = " << mTimeSlices.back()/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation will have  = " << (mTimeSlices.size()-1) << " run(s)\n");
  //GateMessage("Acquisition", 0, "Simulation will generate " << mTotalNbOfParticles << " primaries.\n");

   // Initialize the random engine for the entire simulation
  GateRandomEngine* theRandomEngine = GateRandomEngine::GetInstance();
  theRandomEngine->Initialize();
  if (theRandomEngine->GetVerbosity()>=1) theRandomEngine->ShowStatus();

  GateClock* theClock = GateClock::GetInstance();

/*
  if (!m_pauseFlag) { // skip the initialization if we're coming back from a paused state
    m_time = mTimeSlices[0];
    GateMessage("Geometry", 5, " Start SetTime in GateApplicationMgr before while");
    theClock->SetTime(m_time);
    GateMessage("Geometry", 5, " End SetTime in GateApplicationMgr before while");
  }
*/
  // m_exitFlag = false;
  // m_pauseFlag = false;

  m_virtualStop = mTimeSlices.back();

  if (mOutputMode)
    GateOutputMgr::GetInstance()->RecordBeginOfAcquisition();

  G4int slice=0;
  m_time = mTimeSlices.front();
  while(m_time < mTimeSlices.back()) // && !m_exitFlag && ! m_pauseFlag)
  {
    // Informational message about the current slice
    GateMessage("Acquisition", 0, "Slice " << slice << " from "
                << mTimeSlices[slice]/s << " to "
                << mTimeSlices[slice+1]/s
                << " s [slice="
                << GetTimeInterval(slice)/s
                << " s]\n");

    m_time = mTimeSlices[slice];
    GateMessage("Geometry", 5, " Time is going to change :  = " << m_time/s << Gateendl;);
    theClock->SetTime(m_time);

    // calculate the time steps for total primaries mode
    if(mATotalAmountOfPrimariesIsRequested){
      if(mAnAmountOfPrimariesPerRunIsRequested)
      {
        mTimeStepInTotalAmountOfPrimariesMode = GetTimeInterval(slice)/mRequestedAmountOfPrimariesPerRun;
        m_weight=GetTimeInterval(slice)/(mTimeSlices.back()-mTimeSlices.front());
      }
      else {
        mTimeStepInTotalAmountOfPrimariesMode = (mTimeSlices.back()-mTimeSlices.front())/mRequestedAmountOfPrimaries;
      }
    }

    while(m_time<GetEndTimeSlice(slice))  // sometimes a single slice might require more than MAX_INT events
    {
      GateRunManager::GetRunManager()->SetRunIDCounter(slice); // keep the RunID in sync with the slice #
      GateRunManager::GetRunManager()->BeamOn(INT_MAX);
      theClock->SetTimeNoGeoUpdate(m_time);
    }
    slice++;

  }
  
  if (mOutputMode) GateOutputMgr::GetInstance()->RecordEndOfAcquisition();

  for(int nsource= 0 ; nsource<GateSourceMgr::GetInstance()->GetNumberOfSources() ; nsource++ )
    GateMessage("Acquisition", 1, "Source "<<nsource+1<<" --> Number of events = "<<GateSourceMgr::GetInstance()->GetNumberOfEventBySource(nsource+1)<< Gateendl);

}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StartDAQCluster(G4ThreeVector param) 
{ 
  // With this method we check for all output module enabled but with no
  // filename given. In this case we disable the output module and send a warning.
  GateOutputMgr::GetInstance()->CheckFileNameForAllOutput();

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Source initialization =============\n");
  GateSourceMgr::GetInstance()->Initialization();    // init sources if needed

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Acquisition starts! =============\n");
  InitializeTimeSlices();

  // Verbose
  GateMessage("Acquisition", 0, "Simulation start time = " << mTimeSlices.front()/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation end time   = " << mTimeSlices.back()/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation will have  = " << (mTimeSlices.size()-1) << " run(s)\n");

  // Initialize the random engine for the entire simulation
  GateRandomEngine* theRandomEngine = GateRandomEngine::GetInstance();
  theRandomEngine->Initialize();
  if (theRandomEngine->GetVerbosity()>=1) theRandomEngine->ShowStatus();

  GateClock* theClock = GateClock::GetInstance();

  // check boundary conditions of provided start and stop times
  m_virtualStart= param[0]; // defined from macro from splitter
  m_virtualStop = param[1]; // this one too
  if(m_virtualStart>m_virtualStop)
    GateError("Cluster start time is after cluster stop time.");
  if(m_virtualStart<mTimeSlices.front() || m_virtualStart>mTimeSlices.back())
    GateError("Cluster start time is outside of [StartTime,StopTime]");
  if(m_virtualStop<mTimeSlices.front() || m_virtualStop>mTimeSlices.back())
    GateError("Cluster stop time is outside of [StartTime,StopTime]");
  if (nVerboseLevel>0)
    G4cout << "Cluster: virtual time start " <<m_virtualStart/s<<", virtual time stop "<<m_virtualStop/s<< Gateendl;


  if (mOutputMode) GateOutputMgr::GetInstance()->RecordBeginOfAcquisition();

  G4int slice=0;
  while(m_virtualStart > mTimeSlices[slice+1])
    slice++;

  while(m_time < m_virtualStop) // && !m_exitFlag && ! m_pauseFlag)
  {
    // Informational message about the current slice
    GateMessage("Acquisition", 0, "Slice " << slice << " from "
                << mTimeSlices[slice]/s << " to "
                << mTimeSlices[slice+1]/s
                << " s [slice="
                << GetTimeInterval(slice)/s
                << " s]\n");

    // set the geometry to the beginning of the current slice
    GateMessage("Geometry", 5, " Time is going to change :  = " << m_time/s << Gateendl;);
    theClock->SetTime(mTimeSlices[slice]);

    m_time = max(mTimeSlices[slice],m_virtualStart);
    theClock->SetTimeNoGeoUpdate(m_time);

    // calculate the time steps for total primaries mode
    if(mATotalAmountOfPrimariesIsRequested){
      if(mAnAmountOfPrimariesPerRunIsRequested)
      {
        mTimeStepInTotalAmountOfPrimariesMode = GetTimeInterval(slice)/mRequestedAmountOfPrimariesPerRun;
        m_weight=GetTimeInterval(slice)/(mTimeSlices.back()-mTimeSlices.front());
      }
      else {
        mTimeStepInTotalAmountOfPrimariesMode = (mTimeSlices.back()-mTimeSlices.front())/mRequestedAmountOfPrimaries;
      }
    }

    while(m_time<GetEndTimeSlice(slice))  // sometimes a single slice might require more than MAX_INT events
    {
      GateRunManager::GetRunManager()->SetRunIDCounter(slice); // Must explicitly keep the RunID in sync with the slice #
      GateRunManager::GetRunManager()->BeamOn(INT_MAX);        // otherwise RunID is automatically incremented
      theClock->SetTimeNoGeoUpdate(m_time);
    }
    slice++;

  }
  
  if (mOutputMode) GateOutputMgr::GetInstance()->RecordEndOfAcquisition();

  for(int nsource= 0 ; nsource<GateSourceMgr::GetInstance()->GetNumberOfSources() ; nsource++ )
    GateMessage("Acquisition", 1, "Source "<<nsource+1<<" --> Number of events = "<<GateSourceMgr::GetInstance()->GetNumberOfEventBySource(nsource+1)<< Gateendl);

  return;
  // ========================================================================================================
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StopDAQ() 
{
  m_exitFlag = true;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::PauseDAQ() 
{
  m_pauseFlag = true;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::Describe() 
{
  G4cout << "Data Acquisition summary\n"
	 << "  time slice (s) : " << mTimeSliceDuration/s << Gateendl
	 << "  time start (s) : " << mTimeSlices[0]/s << Gateendl
	 << "  time stop  (s) : " << mTimeSlices[mTimeSlices.size()-1]/s  << Gateendl
	 << "------------------ \n"
	 << Gateendl;
}
//------------------------------------------------------------------------------------------


void GateApplicationMgr::InitializeTimeSlices()
{
  if(mTimeSliceIsSetUsingAddSlice || mTimeSliceIsSetUsingReadSliceInFile)
  {
    // TODO: check that slices are in order
    ;
  }
  else if(mTimeSliceDuration!=0.0)
  {
    G4double endTime = mTimeSlices[1];
    mTimeSlices.resize(1);
    G4int i=0;
    while((mTimeSlices[i] + mTimeSliceDuration) < endTime)
    {
      mTimeSlices.push_back(mTimeSlices[i] + mTimeSliceDuration);
      i++;
    }
    mTimeSlices.push_back(endTime);
  }
  else
    mTimeSliceDuration = mTimeSlices[1]-mTimeSlices[0];

  return;
}

//------------------------------------------------------------------------------------------
/*int GateApplicationMgr::ComputeNumberOfGeneratedPrimaries() {
  GateVSource * source = GateSourceMgr::GetInstance()->GetSource(0);
  int mTotalNbOfParticles = 0;
  for(unsigned int i=0;i<listOfTimeSlice.size();i++) {
    int n = int(rint(listOfTimeSlice[i]*source->GetActivity()));
    //DD(n);
    mTotalNbOfParticles += n;
  }
  return mTotalNbOfParticles;
  }*/
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::EnableTimeStudy(G4String filename)
{
  GateUserActions::GetUserActions()->EnableTimeStudy(filename);
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::EnableTimeStudyForSteps(G4String filename)
{
  GateUserActions::GetUserActions()->EnableTimeStudyForSteps(filename);
}
//------------------------------------------------------------------------------------------

void GateApplicationMgr::PrintStatus()
{
    const G4Run * run = GateRunManager::GetRunManager()->GetCurrentRun();
    const int runID = run->GetRunID() + 1;
    const int runTotal = mTimeSlices.size()-1;

    const int eventID = run->GetNumberOfEvent() + 1;
    int eventTotal = 0;

    if(IsTotalAmountOfPrimariesModeEnabled()) {
        eventTotal = GetTotalNumberOfPrimaries()/runTotal;
    }else if (IsAnAmountOfPrimariesPerRunModeEnabled()) {
        eventTotal = GetNumberOfPrimariesPerRun();
    }else {
        eventTotal = run->GetNumberOfEventToBeProcessed();
    }

    std::clog << "Run ID : " << runID << " / " << runTotal
    << " ; Event ID : " << eventID << " / " << eventTotal
    << " ; " << GetCurrentTime()/CLHEP::s << " / " << GetTimeStop()/CLHEP::s << " s [" << floor(GetCurrentTime()*10000.0/GetTimeStop())/100 << "%]"
    << Gateendl;
}
