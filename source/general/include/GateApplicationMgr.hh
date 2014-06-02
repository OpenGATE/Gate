/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateApplicationMgr_H
#define GateApplicationMgr_H 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "GateConfiguration.h"
#include "GateApplicationMgrMessenger.hh"
#include <vector>

class GateApplicationMgr
{
public:
  static GateApplicationMgr* GetInstance() {
    if (instance == 0)
      instance = new GateApplicationMgr();
    return instance;
  }

  virtual ~GateApplicationMgr();

  G4double GetTimeSlice();
  G4double GetTimeSlice(int run);
  G4double GetEndTimeSlice(int run);
  void SetTimeSlice(G4double value);
  G4double GetTimeStart();
  void SetTimeStart(G4double value);
  G4double GetTimeStop();
  void SetTimeStop(G4double value);

  G4double GetVirtualTimeStop();
  G4double GetVirtualTimeStart();

  inline G4bool GetExitFlag()           { return m_exitFlag; };
  inline void SetExitFlag(G4bool value) { m_exitFlag = value; };

  inline G4bool GetPauseFlag()           { return m_pauseFlag; };
  inline void SetPauseFlag(G4bool value) { m_pauseFlag = value; };

  void StartDAQ();

  void StartDAQCluster(G4ThreeVector param);

  void StartDAQComplete(G4ThreeVector param);
  void StopDAQ();
  void PauseDAQ();

  void Describe();
  void PrintStatus();
  void SetVerboseLevel(G4int value) { nVerboseLevel = value; }

  G4double GetTimeInterval(int i){return listOfTimeSlice[i];}
  void SetTimeInterval(G4double v);
  void SetActivity(G4double v){ listOfActivity.push_back(v);}
  void SetTotalNumberOfPrimaries(double n);
  long int GetTotalNumberOfPrimaries(){return mRequestedAmountOfPrimaries;}
  void SetNumberOfPrimariesPerRun(double n);
  long int GetNumberOfPrimariesPerRun(){return mRequestedAmountOfPrimariesPerRun;}

  void SetNoOutputMode();
  bool GetOutputMode(){return mOutputMode;}
  //void EnableSuccessiveSourceMode(bool t);
  //bool IsSuccessiveSourceModeIsEnabled();
  bool IsTotalAmountOfPrimariesModeEnabled(){return mATotalAmountOfPrimariesIsRequested;}
  bool IsAnAmountOfPrimariesPerRunModeEnabled(){return mRequestedAmountOfPrimariesPerRun;}
  void ReadTimeSlicesInAFile(G4String filename);

  void SetCurrentTime(G4double value){m_time=value;}
  G4double GetCurrentTime(){return m_time;}


  G4double GetTimeStepInTotalAmountOfPrimariesMode(){return mTimeStepInTotalAmountOfPrimariesMode;}
  G4double GetWeight(){return m_weight;}

  void EnableTimeStudy(G4String filename);
  void EnableTimeStudyForSteps(G4String filename);
  long GetRequestedAmountOfPrimariesPerRun() { return mRequestedAmountOfPrimariesPerRun; }

protected:

  GateApplicationMgr();
  static GateApplicationMgr* instance;
  
  G4double m_timeSlice;
  G4double m_timeStart;
  G4double m_timeStop;

  G4double m_virtualStart;
  G4double m_virtualStop;

  G4int nVerboseLevel;

  G4bool m_pauseFlag;
  G4bool m_exitFlag;

  G4double m_time;

  G4double m_weight;

  std::vector<G4double> listOfTimeSlice;
  std::vector<G4double> listOfEndTimeSlice;
  std::vector<G4double> listOfActivity;
  std::vector<G4double> listOfWeight;

  bool mOutputMode;
  //bool mSuccessiveSourceMode;
  bool mTimeSliceIsSet;
  bool mTimeSliceIsSetUsingAddSlice;
  bool mTimeSliceIsSetUsingReadSliceInFile;
  bool mCstTimeSliceIsSet;
  long int mRequestedAmountOfPrimaries;
  bool mATotalAmountOfPrimariesIsRequested;
  long int mRequestedAmountOfPrimariesPerRun;
  bool mAnAmountOfPrimariesPerRunIsRequested;

  void ComputeTimeStop();
  //int ComputeNumberOfGeneratedPrimaries();
  GateApplicationMgrMessenger* m_appMgrMessenger;

  //double m_currentTime;
  double mTimeStepInTotalAmountOfPrimariesMode;

};

#endif

