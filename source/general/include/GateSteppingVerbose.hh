/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateSteppingVerbose
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/

//class GateSteppingVerbose;

#ifndef GATESteppingVerbose_hh
#define GATESteppingVerbose_hh 1

#include "G4SteppingVerbose.hh"

#include "GateInfoForSteppingVerbose.hh"

#include "GateMessageManager.hh"

class G4SliceTimer;

class GateSteppingVerbose : public G4SteppingVerbose 
{
public:   

  GateSteppingVerbose();
  virtual ~GateSteppingVerbose();

  void NewStep();   //beginning of the step
  void StepInfo();  //end of step, before sending informations to Hit/Dig if the volume is sensitive

  void TrackingStarted(); 

  struct GateTrackLevel
  {
	  G4SliceTimer* Timer;
	  G4String Particle;
	  G4String Process;
	  G4String Volume;
	  G4int Energy;
	  G4double Time;
  };
  typedef std::vector<GateTrackLevel> GateTrackLevelVec;

  /*void AtRestDoItInvoked(){}
  void AlongStepDoItAllDone(){}
  void PostStepDoItAllDone(){}
  void AlongStepDoItOneByOne(){}
  void PostStepDoItOneByOne(){}
  void VerboseTrack(){}
  void VerboseParticleChange(){}

  void DPSLStarted(){}
  void DPSLUserLimit(){}
  void DPSLPostStep(){}
  void DPSLAlongStep(){}*/

  void ShowStep() const;

  G4int GetEnergyRange(G4double energy);
  void RecordTrack();
  void EndOfTrack();
  void EndOfRun();
  void EndOfEvent();
  void DisplayTrack(G4String particle,std::ofstream &os);
  void DisplayStep(G4String particle,std::ofstream &os);
  void Initialize(G4String filename, bool stepTracking);

protected:
  G4SliceTimer* pStepTime;
  G4SliceTimer* pTrackTime;
  G4SliceTimer* pTempTime;

  G4String mFileName;
  //G4int currentID;
  //G4bool mNewTrack;
  G4double mSumStepTime;

  GateTrackLevel mTempo;
  G4int currentID;

  bool mIsTrackingStep;
  G4int mNumberOfInit; 
  //std::vector<GateInfoForSteppingVerbose *> theListOfStep; 
 
  G4int currentTrack;

  GateTrackLevelVec theListOfStep;
  GateTrackLevelVec theListOfTrack;

  //run, event?

};

#endif

