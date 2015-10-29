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
  mOutputMode(true),  mTimeSliceIsSetUsingAddSlice(false), mTimeSliceIsSetUsingReadSliceInFile(false)
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

  /* TODO: this does nothing for now. Fix later. JS 28/10/2015
  // Open file  
  std::ifstream is;
  OpenFileInput(filename, is);
  skipComment(is);
  
  // Use Time
  double timeUnit=0.;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << filename << "' need to begin with 'Time'\n");
  }
  
  // Loop line
  skipComment(is);
  int n=0;
  double prevT=0;
  while (is) {
    // Read time
    double t = ReadDouble(is)*timeUnit;
    // Compute slice duration
    if (n == 0) {
      SetTimeStart(t);
    }
    else {
      //listOfEndTimeSlice.push_back(t);
      if (listOfTimeSlice.size() == 0) listOfTimeSlice.push_back(t);// ? non sauf si le premier temps est le depart et le deuxieme le premier intervalle 
      else listOfTimeSlice.push_back(t-prevT);
      if (t<prevT) {
        GateError("Time slices should be in increasing order, but I read " << t/s 
                  << " sec after " << prevT/s << " sec.\n");
        exit(-1);                  
      }
      // DD((t-prevT)/s);
      prevT = t;
    }
    SetTimeStop(t);
    n++;
    skipComment(is);
  }

  // End
  is.close();

  mTimeSliceIsSet = true;
  mTimeSliceIsSetUsingReadSliceInFile = true;
  */
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
  {
    G4double previous_slice = mTimeSlices[mTimeSlices.size()-1];
    mTimeSlices.push_back(previous_slice+v);
  }
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
  return mTimeSlices[run+1];
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
  m_exitFlag = false;
  m_pauseFlag = false;

  if (mOutputMode)
    GateOutputMgr::GetInstance()->RecordBeginOfAcquisition();

  G4int slice=0;
  while(slice<(mTimeSlices.size()-1) && !m_exitFlag && ! m_pauseFlag)
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

    while(m_time<mTimeSlices[slice+1])
    {
      GateRunManager::GetRunManager()->SetRunIDCounter(slice); // otherwise the runID gets incremented each time
      GateRunManager::GetRunManager()->BeamOn(INT_MAX);
      theClock->SetTimeNoGeoUpdate(m_time);
      G4cout << "time: " << m_time/s << G4endl;
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
/*  // With this method we check for all output module enabled but with no
  // filename given. In this case we disable the output module and send a warning.
  GateOutputMgr::GetInstance()->CheckFileNameForAllOutput();

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Source initialization =============\n");

  // Compute timeStop according to time slices
  ComputeTimeStop();

  // init sources if needed
  GateSourceMgr::GetInstance()->Initialization();

  if (listOfTimeSlice.size() == 0)
  {
    GateError("Problem in GateApplicationMgr::StartDAQCluster - No timeslice detected at all !");
  }

  GateMessage("Acquisition", 0,"  \n");
  GateMessage("Acquisition", 0, "============= Acquisition starts! =============\n");

  // Check if start/stop ok
  if (m_timeStop<=m_timeStart)
  {
    GateError("TimeStart is " << m_timeStart/s << " sec, while timeStop is "
              << m_timeStop/s << " sec. Please provide timeStop after timeStart");
  }

  // Verbose
  GateMessage("Acquisition", 0, "Simulation start time = " << m_timeStart/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation end time   = " << m_timeStop/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation will have  = " << listOfTimeSlice.size() << " run(s)\n");

  // It's where we initialize the random engine for the entire simulation
  GateRandomEngine* theRandomEngine = GateRandomEngine::GetInstance();
  theRandomEngine->Initialize();
  if (theRandomEngine->GetVerbosity()>=1) theRandomEngine->ShowStatus();

  GateClock* theClock = GateClock::GetInstance();

  if (!m_pauseFlag) {
    m_time = m_timeStart;
    GateMessage("Geometry", 5, " Start SetTime in GateApplicationMgr before while");
    theClock->SetTime(m_time);
    GateMessage("Geometry", 5, " End SetTime in GateApplicationMgr before while");
  }

  m_exitFlag = false;
  m_pauseFlag = false;


  for (unsigned int i=0; i<listOfTimeSlice.size(); i++) G4cout << "Slice n° " << i << "  | Time: " << listOfTimeSlice[i] << Gateendl;

  // ========================================================================================================
  // It is where the startDAQCluster command differs from the normal startDAQ command

  m_virtualStart= param[0]; // defined from macro from splitter
  m_virtualStop = param[1]; // this one too
  if (nVerboseLevel>0) G4cout << "Cluster: virtual time start " <<m_virtualStart/s<<", virtual time stop "<<m_virtualStop/s<< Gateendl;

  G4double real_timeStop  = m_timeStop;
  m_timeStop=m_virtualStop;

  // we have to go to the start of the current time slice
  // the point before virtual start
  G4double virtualTime = m_timeStart;

  G4int slice=0;
  while (virtualTime<=m_virtualStart)
  {
    virtualTime+=listOfTimeSlice[slice];
    slice++;
  }
  m_time=virtualTime-listOfTimeSlice[slice-1];
  GateRunManager::GetRunManager()->SetRunIDCounter(slice-1);
  theClock->SetTime(m_time);
  if (nVerboseLevel>0) G4cout << "Cluster: time start for geometry" <<m_time<<", runID moved to "<<slice-1<< Gateendl;

  if (mOutputMode) GateOutputMgr::GetInstance()->RecordBeginOfAcquisition();

  while ((m_time < m_timeStop) && (!m_exitFlag) && (!m_pauseFlag))
  {
    m_timeSlice=listOfTimeSlice[slice-1];
    if(IsTotalAmountOfPrimariesModeEnabled())
    {
      if(!mAnAmountOfPrimariesPerRunIsRequested) mTimeStepInTotalAmountOfPrimariesMode = (real_timeStop-m_timeStart)/mRequestedAmountOfPrimaries;
      else
      {
          mTimeStepInTotalAmountOfPrimariesMode = m_timeSlice/mRequestedAmountOfPrimariesPerRun;
          m_weight=m_timeSlice/(real_timeStop-m_timeStart);
      }
    }

    if(m_time<m_virtualStart)
    {
        // we move to the virtualStart without geometry update
        // and change to a smaller timeSlice accordingly
        m_timeSlice=m_time+listOfTimeSlice[slice-1]-m_virtualStart;
        m_time = m_virtualStart;
        theClock->SetTimeNoGeoUpdate(m_time);
    }
    if(m_time+listOfTimeSlice[slice-1]>m_virtualStop)
    {
       // we have to stop before the end of the timeSlice
       // so we reduce m_timeSlice to match with  m_virtualStop
       m_timeSlice=m_virtualStop-m_time;
    }
    GateRunManager::GetRunManager()->BeamOn(INT_MAX);
    slice++;
    theClock->SetTime(m_time);
  }
  m_timeStop=real_timeStop;
  
  if (mOutputMode) GateOutputMgr::GetInstance()->RecordEndOfAcquisition();

  for(int nsource= 0 ; nsource<GateSourceMgr::GetInstance()->GetNumberOfSources() ; nsource++ )
    GateMessage("Acquisition", 1, "Source "<<nsource+1<<" --> Number of events = "<<GateSourceMgr::GetInstance()->GetNumberOfEventBySource(nsource+1)<< Gateendl);
*/
  return;
  // ========================================================================================================
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StopDAQ() 
{
  SetExitFlag(true);
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::PauseDAQ() 
{
  SetPauseFlag(true);
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
  else if(mTimeSliceDuration!=0.0) // could check that TimeSliceDuration is non-zero
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
