/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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

  void StartDAQ();
  void StartDAQCluster(G4ThreeVector param);

  void StartDAQComplete(G4ThreeVector param);
  void StopDAQ() {};
  void PauseDAQ() {};

  void Describe();
  void PrintStatus();
  void SetVerboseLevel(G4int value) { nVerboseLevel = value; }

  void SetTimeInterval(G4double v);

  void SetTotalNumberOfPrimaries(double n);
  long int GetTotalNumberOfPrimaries(){return mRequestedAmountOfPrimaries;}
  
  void SetNumberOfPrimariesPerRun(double n);
  long int GetNumberOfPrimariesPerRun(){return mRequestedAmountOfPrimariesPerRun;}

  //LSLS
  void ReadNumberOfPrimariesInAFile(G4String filename);
  bool IsReadNumberOfPrimariesInAFileModeEnabled(){return mReadNumberOfPrimariesInAFileIsUsed;}



  void SetNoOutputMode();
  bool GetOutputMode(){return mOutputMode;}
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
  
  G4double m_clusterStart;
  G4double m_clusterStop;

  G4int nVerboseLevel;

  G4double m_time;

  G4double m_weight;

  G4double mTimeSliceDuration;
  std::vector<G4double> mTimeSlices;
  
  //LSLS
  std::vector<G4int> mNumberOfPrimariesPerRun;
  bool mReadNumberOfPrimariesInAFileIsUsed;



  bool mOutputMode;
  bool mTimeSliceIsSetUsingAddSlice;
  bool mTimeSliceIsSetUsingReadSliceInFile;

  long int mRequestedAmountOfPrimaries;
  bool mATotalAmountOfPrimariesIsRequested;
  long int mRequestedAmountOfPrimariesPerRun;
  bool mAnAmountOfPrimariesPerRunIsRequested;
  double mTimeStepInTotalAmountOfPrimariesMode;

  void InitializeTimeSlices();

  GateApplicationMgrMessenger* m_appMgrMessenger;

};

#endif

