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
  m_timeSlice(1*s),  m_timeStart(0*s),  m_timeStop(1*s), 
  nVerboseLevel(0), m_pauseFlag(false), m_exitFlag(false), 
  mOutputMode(true), mTimeSliceIsSet(false),
  mTimeSliceIsSetUsingAddSlice(false), mTimeSliceIsSetUsingReadSliceInFile(false),
  mCstTimeSliceIsSet(false)
{
  if(instance != 0)
    { G4Exception( "GateApplicationMgr::GateApplicationMgr", "GateApplicationMgr", FatalException, "GateApplicationMgr constructed twice."); }
  m_appMgrMessenger = new GateApplicationMgrMessenger();

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
  if (mCstTimeSliceIsSet) {
    GateError("Please do not use 'addSlice' or 'readTimeSlicesIn' commands with 'setTimeSlice' command");
  }
  if (mTimeSliceIsSetUsingAddSlice) {
    GateError("Please do not use 'addSlice' and 'readTimeSlicesIn' commands at the same time");
  }
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
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeSlice(G4double timeSlice) 
{
  if (mTimeSliceIsSet) {
    GateError("Please do not use 'setTimeSlice' command with 'addTimeSlice' or 'readTimeSlicesIn' commands");
  }
  mCstTimeSliceIsSet = true;
  m_timeSlice = timeSlice;
  //if (nVerboseLevel>0) G4cout << "Time Slice set to (s) " << m_timeSlice/s << G4endl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeInterval(G4double v)
{
  if (mCstTimeSliceIsSet) {
    GateError("Please do not use 'addSlice' or 'readTimeSlicesIn' commands with 'setTimeSlice' command");
  }
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateError("Please do not use 'addSlice' and 'readTimeSlicesIn' commands at the same time");
  }
  mTimeSliceIsSet = true;
  mTimeSliceIsSetUsingAddSlice = true;

  listOfTimeSlice.push_back(v);
  //if (nVerboseLevel>0) G4cout << "Time Slice set to (s) " << m_timeSlice/s << G4endl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeSlice() 
{
  return m_timeSlice;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeSlice(int run) 
{
  //if(listOfTimeSlice.size()==0) return m_timeSlice;

  if (run>=(int)listOfTimeSlice.size()) {
    GateWarning("Warning in GateApplicationMgr::GetTimeSlice, run=" << run << " is greater than the list of slices. Do nothing\n");
    return 0.0; // DO NOTHING in this case
  }
  return listOfTimeSlice[run];
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetEndTimeSlice(int run) 
{
  //if(listOfTimeSlice.size()==0) return m_timeSlice;

  if (run>=(int)listOfEndTimeSlice.size()) {
    GateError("Error in GateApplicationMgr::GetTimeSlice, run=" << run << "\n");
  }
  return listOfEndTimeSlice[run];
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeStart(G4double timeStart) 
{
  if (timeStart<0.0) {
    GateError("setTimeStart shoud not be negative");
  }
  if (mTimeSliceIsSetUsingReadSliceInFile) {
    GateError("setTimeStart command cannot be used with readTimeSlicesIn command. The first time in file is taken as timeStart.");
  }
  m_timeStart = timeStart;
  if (nVerboseLevel>0) G4cout << "Time Start set to (s) " << m_timeStart/s << G4endl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeStart() 
{
  return m_timeStart;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::SetTimeStop(G4double timeStop) 
{
  if (timeStop<0.0) {
    GateError("setTimeStop shoud not be negative");
  }
  m_timeStop = timeStop;
  if (nVerboseLevel>0) G4cout << "Time Stop set to (s) " << m_timeStop/s << G4endl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
G4double GateApplicationMgr::GetTimeStop() 
{
  return m_timeStop;
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

  GateMessage("Acquisition", 0,"                                     \n");
  GateMessage("Acquisition", 0, "============= Source initialization =============\n");
 
  // Compute timeStop according to time slices
  ComputeTimeStop();

  // init sources if needed
  GateSourceMgr::GetInstance()->Initialization();

  if (listOfTimeSlice.size() == 0) {
    SetTimeStart(0.0*s);
    SetTimeStop(1.0*s);
    listOfTimeSlice.push_back(1*s);
  }
  double time = GetTimeStart();
  for(size_t j = 0 ; j<listOfTimeSlice.size() ; j++){
    time += listOfTimeSlice[j];
    listOfEndTimeSlice.push_back(time);
  }

  GateMessage("Acquisition", 0,"                                     \n");
  GateMessage("Acquisition", 0, "============= Acquisition starts! =============\n");

  // Check if start/stop ok
  if (m_timeStop<m_timeStart) {
    GateError("TimeStart is " << m_timeStart/s << " sec, while timeStop is "
              << m_timeStop/s << " sec. Please provide timeStop after timeStart");
  }
  
  // Check if start and stop is equal : assume a single run in this case
  if (m_timeStop == m_timeStart) { // assume a single run
    //    GateWarning("Time start is equal to time stop. I assume a single run from 0 to 1 sec.\n");
    SetTimeStart(0.0*s);
    SetTimeStop(1.0*s);
    if (mTimeSliceIsSet || mCstTimeSliceIsSet) {
      GateWarning("Your time slices will be ignored.");
    }
    mTimeSliceIsSet = false;
    mCstTimeSliceIsSet = false;
    SetTimeSlice(1.0*s);
  }
   
  // Compute timeStop according to time slices
  //ComputeTimeStop();

  // Set the exact number
  /*double mTotalNbOfParticles = ComputeNumberOfGeneratedPrimaries();
  GateVSource * source = GateSourceMgr::GetInstance()->GetSource(0);
  if (mRequestedAmountOfPrimaries != mTotalNbOfParticles) {
    double diff = mRequestedAmountOfPrimaries - mTotalNbOfParticles;
    double initialLastSlice = listOfTimeSlice.back();
    double x = diff/source->GetActivity()/s;
    int lastN = rint(listOfTimeSlice.back()*source->GetActivity());
    int newLastN = lastN;
    while (newLastN != lastN+diff) {
      listOfTimeSlice.back() = listOfTimeSlice.back() + x*s/2;
      newLastN = rint(listOfTimeSlice.back()*source->GetActivity());
    }
    ComputeTimeStop();      
    mTotalNbOfParticles = ComputeNumberOfGeneratedPrimaries();
    GateWarning("I slightly change the last slice, from " 
                << initialLastSlice/s << " sec to "
                << listOfTimeSlice.back()/s << " sec, to reach "
                << mTotalNbOfParticles << " primaries. TimeStop is now " << GetTimeStop()/s
                << std::endl);
  }
  */

  // If needed, compute the source activity to reach a total amount of particle
  //double mTotalNbOfParticles = ComputeNumberOfGeneratedPrimaries();
  //DD(mTotalNbOfParticles);
  /*if (mATotalAmountOfPrimariesIsRequested) {
    if (mSuccessiveSourceMode) {
      GateVSource * source = GateSourceMgr::GetInstance()->GetSource(0);
      double activity = mRequestedAmountOfPrimaries/((GetTimeStop()-GetTimeStart())/s);
      //DD(activity);
      //DD(source->GetActivity());
      source->SetActivity(activity/s);
      //DD(source->GetActivity()); 
      GateMessage("Acquisition", 0, "Simulation activity changed to = " << activity/s << " sec.\n");      
      mTotalNbOfParticles = ComputeNumberOfGeneratedPrimaries();

      // Set the exact number
      if (mRequestedAmountOfPrimaries != mTotalNbOfParticles) {
        double diff = mRequestedAmountOfPrimaries - mTotalNbOfParticles;
        double initialLastSlice = listOfTimeSlice.back();
        double x = diff/source->GetActivity()/s;
        int lastN = rint(listOfTimeSlice.back()*source->GetActivity());
        int newLastN = lastN;
        while (newLastN != lastN+diff) {
          listOfTimeSlice.back() = listOfTimeSlice.back() + x*s/2;
          newLastN = rint(listOfTimeSlice.back()*source->GetActivity());
        }
        ComputeTimeStop();      
        mTotalNbOfParticles = ComputeNumberOfGeneratedPrimaries();

        GateWarning("I slightly change the last slice, from " 
                    << initialLastSlice/s << " sec to "
                    << listOfTimeSlice.back()/s << " sec, to reach "
                    << mTotalNbOfParticles << " primaries. TimeStop is now " << GetTimeStop()/s
                    << std::endl);
      }
    }
  }*/

 /* if (mATotalAmountOfPrimariesIsRequested) {
    if (mRequestedAmountOfPrimaries != mTotalNbOfParticles) {
      GateError("Requested nb of primaries is " << mRequestedAmountOfPrimaries
                << " but planned is " << mTotalNbOfParticles << G4endl);
    }
  }*/

  // Verbose 
  GateMessage("Acquisition", 0, "Simulation start time = " << m_timeStart/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation end time   = " << m_timeStop/s << " sec\n");
  GateMessage("Acquisition", 0, "Simulation will have  = " << listOfTimeSlice.size() << " run(s)\n");
  //GateMessage("Acquisition", 0, "Simulation will generate " << mTotalNbOfParticles << " primaries.\n");

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

 /* if (m_timeSlice > m_timeStop - m_timeStart) {
    if (nVerboseLevel>0) G4cout << "WARNING: Time Slice bigger than DAQ total time" << G4endl;
  }*/

  m_exitFlag = false;
  m_pauseFlag = false;

  if (mOutputMode) GateOutputMgr::GetInstance()->RecordBeginOfAcquisition();

  G4int slice=0;
  while ((m_time < m_timeStop) && (!m_exitFlag) && (!m_pauseFlag)) {
    //  GateMessage("Acquisition", 0, G4endl);

    double time1 = GetTimeStart();
    double time2 = 0.;
    if(slice>0) time1 = listOfEndTimeSlice[slice-1];
    time2 = listOfEndTimeSlice[slice];
    
    GateMessage("Acquisition", 0, "Slice " << slice << " from " 
                << time1/s << " to " 
                << time2/s 
                << " s [slice=" 
                << listOfTimeSlice[slice]/s
                << " s], final stop at " << m_timeStop/s << " s.\n");    

    m_timeSlice=listOfTimeSlice[slice];    
    //GateMessage("Acquisition", 0, "Current slice is " << m_timeSlice/s << "\n");

    if(IsTotalAmountOfPrimariesModeEnabled()){
      if(!mAnAmountOfPrimariesPerRunIsRequested) mTimeStepInTotalAmountOfPrimariesMode = (m_timeStop-m_timeStart)/mRequestedAmountOfPrimaries;
      else {
          mTimeStepInTotalAmountOfPrimariesMode = m_timeSlice/mRequestedAmountOfPrimariesPerRun;
          m_weight=m_timeSlice/(m_timeStop-m_timeStart);
      }
    } 
    GateRunManager::GetRunManager()->BeamOn(INT_MAX);
    //m_time += listOfTimeSlice[slice];//m_timeSlice;
    //GateMessage("Acquisition", 0, "Slice after time = " << m_time/s << "\n");

    GateMessage("Geometry", 5, " Time is going to be change :  = " << m_time/s << G4endl;);
    slice++;
    theClock->SetTime(m_time);

    //GateMessage("Geometry", 0, "Change geom status !!!" << G4endl);
    //GateDetectorConstruction::GetGateDetectorConstruction()->SetGeometryStatusFlag(GateDetectorConstruction::geometry_needs_rebuild);
  }
  
  if (mOutputMode) GateOutputMgr::GetInstance()->RecordEndOfAcquisition();


  for(int nsource= 0 ; nsource<GateSourceMgr::GetInstance()->GetNumberOfSources() ; nsource++ )
    GateMessage("Acquisition", 1, "Source "<<nsource+1<<" --> Number of events = "<<GateSourceMgr::GetInstance()->GetNumberOfEventBySource(nsource+1)<<G4endl);

}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::StartDAQCluster(G4ThreeVector param) 
{ 
  // With this method we check for all output module enabled but with no
  // filename given. In this case we disable the output module and send a warning.
  GateOutputMgr::GetInstance()->CheckFileNameForAllOutput();

  GateMessage("Acquisition", 0,"                                     \n");
  GateMessage("Acquisition", 0, "============= Source initialization =============\n");

  // Compute timeStop according to time slices
  ComputeTimeStop();

  // init sources if needed
  GateSourceMgr::GetInstance()->Initialization();

  if (listOfTimeSlice.size() == 0)
  {
    GateError("Problem in GateApplicationMgr::StartDAQCluster - No timeslice detected at all !");
  }
  double time = GetTimeStart();
  for(size_t j = 0 ; j<listOfTimeSlice.size() ; j++){
    time += listOfTimeSlice[j];
    listOfEndTimeSlice.push_back(time);
  }

  GateMessage("Acquisition", 0,"                                     \n");
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


  for (unsigned int i=0; i<listOfTimeSlice.size(); i++) G4cout << "Slice n° " << i << "  | Time: " << listOfTimeSlice[i] << G4endl;

  // ========================================================================================================
  // It is where the startDAQCluster command differs from the normal startDAQ command

  m_virtualStart= param[0]; // defined from macro from splitter
  m_virtualStop = param[1]; // this one too
  if (nVerboseLevel>0) G4cout << "Cluster: virtual time start " <<m_virtualStart/s<<", virtual time stop "<<m_virtualStop/s<<G4endl;

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
  if (nVerboseLevel>0) G4cout << "Cluster: time start for geometry" <<m_time<<", runID moved to "<<slice-1<<G4endl;

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
    GateMessage("Acquisition", 1, "Source "<<nsource+1<<" --> Number of events = "<<GateSourceMgr::GetInstance()->GetNumberOfEventBySource(nsource+1)<<G4endl);

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
  G4cout << "Data Acquisition summary" << G4endl
	 << "  time slice (s) : " << m_timeSlice/s << G4endl
	 << "  time start (s) : " << m_timeStart/s << G4endl
	 << "  time stop  (s) : " << m_timeStop/s  << G4endl
	 << "------------------ " << G4endl
	 << G4endl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateApplicationMgr::ComputeTimeStop()
{
  if(listOfTimeSlice.size()>0)
  {
//    if(m_timeStop!=0) GateError("Please do not use 'setTimeStop' command with 'addTimeSlice' or 'readTimeSlicesIn' commands");
    m_timeStop = m_timeStart;
    for(unsigned int i=0;i<listOfTimeSlice.size();i++) {
      m_timeStop += listOfTimeSlice[i];
    }
  }
  else {
    if (m_timeSlice > m_timeStop - m_timeStart) {
       GateWarning("Time Slice bigger than DAQ total time");
    }
    int n = int((m_timeStop-m_timeStart)/m_timeSlice);
    for(int i=0;i<n;i++) listOfTimeSlice.push_back(m_timeSlice);
    if (n*m_timeSlice < m_timeStop) listOfTimeSlice.push_back(m_timeStop-n*m_timeSlice);
  }

  double time = GetTimeStart();
  for(size_t j = 0 ; j<listOfTimeSlice.size() ; j++){
    time += listOfTimeSlice[j];
    listOfEndTimeSlice.push_back(time);
  }

}
//------------------------------------------------------------------------------------------


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
    const G4Run * run = G4RunManager::GetRunManager()->GetCurrentRun();
    const int runID = run->GetRunID() + 1;
    const int runTotal = listOfTimeSlice.size();

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
    << std::endl;
}
