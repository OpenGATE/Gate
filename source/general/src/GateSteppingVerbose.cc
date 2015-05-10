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


#include "GateSteppingVerbose.hh"
//#include "G4SteppingManager.hh"
#include "GateMiscFunctions.hh"

//#include "G4VSensitiveDetector.hh"    // Include from 'hits/digi'
//#include "G4StepStatus.hh"    // Include from 'tracking'

#include "G4SystemOfUnits.hh"
#include "G4SliceTimer.hh"
#include "G4TouchableHistory.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManagerKernel.hh"
#include "G4TrackingManager.hh"

#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
#include "G4UnitsTable.hh"
#else
#define G4BestUnit(a,b) a
#endif


//==================================================
GateSteppingVerbose::GateSteppingVerbose():G4SteppingVerbose(), mSumStepTime(0)
{
  pStepTime = new G4SliceTimer;
  pTrackTime = new G4SliceTimer;
  pStepTime->Clear();

  mTempo.Energy = 0.;
  mTempo.Process = "";
  mTempo.Particle = "";
  mTempo.Volume = "";
  currentID=-1;

  currentTrack = -1;
  pTempTime = 0;

  mFileName = "";
  mIsTrackingStep = false;
  mNumberOfInit = 0;
  //mNewTrack = true;
  //mTrackTimeTempo = 0.;
  // Set verbosity for timing
                
}
//==================================================


//==================================================
GateSteppingVerbose::~GateSteppingVerbose()
{
  if(pStepTime)  delete pStepTime;
  if(pTrackTime) delete pTrackTime;
  if(pTempTime) delete pTempTime;

  for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); )
	  it=theListOfStep.erase(it);
  for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); )
	  it=theListOfTrack.erase(it);

}
//==================================================


//==================================================
void GateSteppingVerbose::NewStep()
{
  if(!mIsTrackingStep)G4SteppingVerbose::NewStep();
  if(!mIsTrackingStep) return

  CopyState();

  if(mTempo.Volume != fTrack->GetVolume()->GetName())
  { 
    theListOfTrack[currentTrack].Timer->Stop();
    mTempo.Volume = fTrack->GetVolume()->GetName();

    bool knownState = false;

    for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
    {
      if(it->Particle == mTempo.Particle && it->Process == mTempo.Process &&
      it->Volume == mTempo.Volume && it->Energy == mTempo.Energy)
      {
        if(knownState) GateError("This state was already found.");
        it->Timer->Start();// += pTrackTime->GetUserElapsed();
        currentTrack = it-theListOfTrack.begin();
        knownState=true;
      }
    }

    if(!knownState)
    {
      mTempo.Timer = new G4SliceTimer;
      theListOfTrack.push_back(mTempo);
      currentTrack = theListOfTrack.size()-1;
      theListOfTrack[currentTrack].Timer->Start();
    }

    
  } 

  if(pTempTime) delete pTempTime;
  pTempTime = 0;
  pTempTime = new G4SliceTimer();

  pStepTime->Start(); 
  pTempTime->Start(); 
}
//==================================================


//==================================================
void GateSteppingVerbose::StepInfo()
{
  if(!mIsTrackingStep)G4SteppingVerbose::StepInfo();
  if(!mIsTrackingStep) return;

  //if(!pTrackTime->IsValid()) pTrackTime->Stop();
 // CopyState();
  /*for(unsigned int i = 0; i<theListOfParticle.size();i++)
  {
   if(!theListOfTimer[i]->IsValid()) theListOfTimer[i]->Stop();
  }*/

  if(!pStepTime->IsValid()) pStepTime->Stop();
  pTempTime->Stop();

  CopyState();

  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(fStep->GetPreStepPoint()->GetTouchable());
  G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();
  GateTrackLevel StepLevel;
  StepLevel.Particle = fStep->GetTrack()->GetDefinition()->GetParticleName();
  StepLevel.Volume = currentVol->GetName();
  StepLevel.Process = fCurrentProcess ? fCurrentProcess->GetProcessName() : "NoProcess";
  StepLevel.Energy = GetEnergyRange(fStep->GetPreStepPoint()->GetKineticEnergy());
  bool knownState = false;

  for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
  {
      if(it->Particle == StepLevel.Particle && it->Process ==  StepLevel.Process &&
    	  it->Volume == StepLevel.Volume && it->Energy == StepLevel.Energy)
      {
        if(knownState) GateError("This state was already found.");
        it->Time += pTempTime->GetUserElapsed();
        //theListOfTimer[i]->Start();// += pStepTime->GetUserElapsed();
        knownState=true;
      }
  }

  if(!knownState)
  {
      //G4cout<<mTempoParticle<<"  "<<mTempoProcess<<"  "<<theListOfProcess.size()<<"  "<<theListOfParticle.size()<< Gateendl;
	  StepLevel.Time = pTempTime->GetUserElapsed();
	  theListOfStep.push_back(StepLevel);
  }

}
//==================================================


//==================================================
void GateSteppingVerbose::TrackingStarted()
{
  if(!mIsTrackingStep)G4SteppingVerbose::TrackingStarted();
  CopyState();

  mTempo.Energy   = GetEnergyRange(fTrack->GetKineticEnergy());
  if( fTrack->GetCreatorProcess()) mTempo.Process  = fTrack->GetCreatorProcess()->GetProcessName();
  else mTempo.Process  = "Source";
  mTempo.Particle = fTrack->GetDefinition()->GetParticleName();
  mTempo.Volume   = fTrack->GetVolume()->GetName();//GetLogicalVolumeAtVertex()->GetName();

  bool knownState = false;

  for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
  {
    if(it->Particle == mTempo.Particle && it->Process == mTempo.Process &&
    it->Volume == mTempo.Volume && it->Energy == mTempo.Energy)
    {
      if(knownState) GateError("This state was already found.");
      it->Timer->Start();// += pTrackTime->GetUserElapsed();
      currentTrack = it-theListOfTrack.begin();
      knownState=true;
    }
  }

  if(!knownState)
  {
	mTempo.Timer = new G4SliceTimer;
	theListOfTrack.push_back(mTempo);
    currentTrack = theListOfTrack.size()-1;
    theListOfTrack[currentTrack].Timer->Start();
  }

  pTrackTime->Clear();
  pTrackTime->Start();
}
//==================================================


//==================================================
void GateSteppingVerbose::ShowStep() const
{
  if(!mIsTrackingStep)G4SteppingVerbose::ShowStep();
}
//==================================================


//==================================================
G4int GateSteppingVerbose::GetEnergyRange(G4double energy)
{
  //G4int ref = 6; // Energy is in MeV ie 10^6 eV
  G4int min = 3; // 1 keV (10^3)
  
  G4int range = min;
  G4double tempoEnergy = energy/keV;
  while(tempoEnergy>=10)
  {
    tempoEnergy /= 10.;
    range++;
  }

  return range;
}
//==================================================


//==================================================
void GateSteppingVerbose::RecordTrack()
{
   if(!pTrackTime->IsValid()) pTrackTime->Stop();    
  theListOfTrack[currentTrack].Timer->Stop();

}
//==================================================


//==================================================
void GateSteppingVerbose::EndOfTrack()
{
  RecordTrack();
}
//==================================================


//==================================================
void GateSteppingVerbose::EndOfRun()
{


 /* for(unsigned int i = 0; i<theListOfParticle.size();i++)
  {
    G4cout<<theListOfParticle[i]<<"     "<<theListOfTimer[i]->GetUserElapsed()<<"   "<<G4BestUnit(theListOfTimer[i]->GetRealElapsed() , "Time")<< Gateendl;
//G4BestUnit(theListOfUsedTime[i] , "Time");
  }*/

  GateTrackLevelVec UsedThings;

  bool alreadyUsed = false;

  for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
  {
     alreadyUsed = false;
     for(GateTrackLevelVec::iterator jt = UsedThings.begin(); jt!=UsedThings.end(); jt++)
     {
        if(it->Particle == jt->Particle)
        {
          jt->Time +=  it->Timer->GetUserElapsed();
          alreadyUsed = true;
        }
     }
     if(!alreadyUsed) 
     {
       GateTrackLevel notused;
       notused.Particle=it->Particle;
       notused.Time=it->Timer->GetUserElapsed();
       UsedThings.push_back(notused);
     }
  }

  std::ofstream os;
  OpenFileOutput(mFileName, os);
  os<< Gateendl;
   if(mIsTrackingStep) os<<" Total time elapsed during physical part of steps = "<<pStepTime->GetUserElapsed()<<" s\n";

  os<< Gateendl;
  os<< Gateendl;
  os<<"----> Track level\n";
  os<< Gateendl;
  for(GateTrackLevelVec::iterator jt = UsedThings.begin(); jt!=UsedThings.end(); jt++)
  {
    os<<"------------------------------------------------------------------------\n";
    os<<jt->Particle<<"     "<<jt->Time<<" s\n";
    DisplayTrack(jt->Particle, os);
    os<< Gateendl;
    os<<"------------------------------------------------------------------------\n";
    os<< Gateendl;
    os<< Gateendl;
 }

  alreadyUsed = false;

  UsedThings.clear();

  for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
  {
     //os<<theListOfParticle[i]<<"  "<<theListOfProcess[i]<< Gateendl;
     alreadyUsed = false;
     for(GateTrackLevelVec::iterator jt = UsedThings.begin(); jt!=UsedThings.end(); jt++)
     {
        if(it->Particle==jt->Particle)
        {
          jt->Time +=  it->Time;
          //theListOfUsedTime[j] +=  theListOfTimer[i]->GetUserElapsed();
          alreadyUsed = true;
        }
     }
     if(!alreadyUsed) 
     {
       GateTrackLevel notused;
       notused.Particle=it->Particle;
       notused.Time=it->Time;
       UsedThings.push_back(notused);
     }
  }

 
  os<< Gateendl;
  if(mIsTrackingStep) {
    os<< Gateendl;
    os<<"----> Physical step level\n";
    os<< Gateendl;

    for(GateTrackLevelVec::iterator jt = UsedThings.begin(); jt!=UsedThings.end(); jt++)
    {
      os<<"------------------------------------------------------------------------\n";
      os<<jt->Particle<<"     "<<jt->Time<<" s\n";
      DisplayStep(jt->Particle, os);
      os<< Gateendl;
      os<<"------------------------------------------------------------------------\n";
      os<< Gateendl;
      os<< Gateendl;


      //G4BestUnit(theListOfUsedTime[i] , "Time");
    }
  }
  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mFileName << Gateendl);
  }
  os.flush();
  os.close();

}
//==================================================


//==================================================
void GateSteppingVerbose::EndOfEvent()
{
}
//==================================================

//==================================================
void GateSteppingVerbose::DisplayTrack(G4String particle,std::ofstream &os)
{


  std::map<G4String,G4double> theListOfUsedVolume; 
  std::map<G4String,G4double> theListOfUsedProcess; 

  std::map<G4String,G4double> theListOfTotalPerVolume; 
  std::map<G4String,G4double> theListOfTotalPerProcess; 
  //std::vector<G4double> theListOfUsedVolumeTime; 
  //std::map<G4int,G4String> theListOfUsedProcess; 
  //std::map<G4int,G4double> theListOfUsedProcessTime; 
  //std::vector<G4int> theListOfUsedEnergy;

  G4int energyMax = 0;

  for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
  {
	//Energy
    if(it->Energy > energyMax) energyMax = it->Energy;
    //Volume
    theListOfUsedVolume[it->Volume] = 0.;
    theListOfTotalPerVolume[it->Volume] = 0.;
    //Process
    theListOfUsedProcess[it->Process] = 0.;
    theListOfTotalPerProcess[it->Process] = 0.;
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; it++)
  {
    os<<"\t"<<it->first;
  }
  os<< Gateendl;
 

  for(int j = 3; j<=energyMax;j++)
  {
	for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
    {
      theListOfUsedVolume[it->Volume] = 0.;
      if(it->Particle == particle)
      {
        if(j==it->Energy)
        {
          theListOfUsedVolume[it->Volume] += it->Timer->GetUserElapsed();
          theListOfTotalPerVolume[it->Volume] += it->Timer->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; it++)
    {
      os<<"\t"<<it->second;
    }
    os<< Gateendl;
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerVolume.begin() ; it!=theListOfTotalPerVolume.end() ; it++)
  {
    os<<"\t"<<it->second;
  }
    os<< Gateendl;

    os<< Gateendl;    

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; it++)
  {
    os<<"\t"<<it->first;
  }
  os<< Gateendl;    
 

  for(int j = 3; j<=energyMax;j++)
  {
	for (GateTrackLevelVec::iterator it=theListOfTrack.begin(); it!=theListOfTrack.end(); it++)
    {
      if(it->Particle == particle)
      {
        theListOfUsedProcess[it->Particle] = 0.;
        if(j==it->Energy)
        {
          theListOfUsedProcess[it->Process] += it->Timer->GetUserElapsed();
	      theListOfTotalPerProcess[it->Process] += it->Timer->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; it++)
    {
      os<<"\t"<<it->second;
    }
    os<< Gateendl;   
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerProcess.begin() ; it!=theListOfTotalPerProcess.end() ; it++)
  {
    os<<"\t"<<it->second;
  }
    os<< Gateendl;


}
//==================================================


//==================================================
void GateSteppingVerbose::DisplayStep(G4String particle,  std::ofstream &os)
{

  std::map<G4String,G4double> theListOfUsedVolume; 
  std::map<G4String,G4double> theListOfUsedProcess; 

  std::map<G4String,G4double> theListOfTotalPerVolume; 
  std::map<G4String,G4double> theListOfTotalPerProcess; 
  //std::vector<G4double> theListOfUsedVolumeTime; 
  //std::map<G4int,G4String> theListOfUsedProcess; 
  //std::map<G4int,G4double> theListOfUsedProcessTime; 
  //std::vector<G4int> theListOfUsedEnergy;

  G4int energyMax = 0;

  for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
  {
	//Energy
    if(it->Energy>energyMax) energyMax = it->Energy;
    //Volume
    theListOfUsedVolume[it->Volume] = 0.;
    theListOfTotalPerVolume[it->Volume] = 0.;
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; it++)
  {
    os<<"\t"<<it->first;
  }
  os<< Gateendl;
 

  for(int j = 3; j<=energyMax;j++)
  {
	for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
    {
      theListOfUsedVolume[it->Volume] = 0.;
      if(it->Particle == particle)
      {
        if(j==it->Energy)
        {
          theListOfUsedVolume[it->Volume] += it->Time;
          theListOfTotalPerVolume[it->Volume] += it->Time;
          //theListOfUsedVolume[theListOfVolume[i]] += theListOfTimer[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; it++)
    {
      os<<"\t"<<it->second;
    }
    os<< Gateendl;    
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerVolume.begin() ; it!=theListOfTotalPerVolume.end() ; it++)
  {
    os<<"\t"<<it->second;
  }
    os<< Gateendl;

    os<< Gateendl;    

//Process
  for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
  {
   // os<<theListOfParticle[i]<<"  "<<theListOfProcess[i]<< Gateendl;
    if(it->Particle == particle) {
      theListOfUsedProcess[it->Process] = 0.;
      theListOfTotalPerProcess[it->Process] = 0.;
    }
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; it++)
  {
    os<<"\t"<<it->first;
  }
  os<< Gateendl;    
 

  for(int j = 3; j<=energyMax;j++)
  {
	for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
    {
      if(it->Particle == particle) theListOfUsedProcess[it->Process] = 0.;
    }  
	for (GateTrackLevelVec::iterator it=theListOfStep.begin(); it!=theListOfStep.end(); it++)
    {
      if(it->Particle == particle)
      {
        if(j==it->Energy)
        {
          theListOfUsedProcess[it->Process] += it->Time;
          theListOfTotalPerProcess[it->Process]+= it->Time;
          //theListOfUsedProcess[theListOfProcess[i]] += theListOfTimer[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; it++)
    {
      os<<"\t"<<it->second;
    }
    os<< Gateendl;   
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerProcess.begin() ; it!=theListOfTotalPerProcess.end() ; it++)
  {
    os<<"\t"<<it->second;
  }
    os<< Gateendl;

}
//==================================================


//==================================================
void GateSteppingVerbose::Initialize(G4String filename, bool stepTracking)
{
  if(mNumberOfInit!=0) GateError("Commands '/gate/application/enableStepAndTrackTimeStudy' and '/gate/application/enableTrackTimeStudy' must not be called both in the same simulation or several times in a simulation!");
  mIsTrackingStep = stepTracking;
  if(!mIsTrackingStep)G4RunManagerKernel::GetRunManagerKernel()->GetTrackingManager()->SetVerboseLevel(1);
  if(mIsTrackingStep)G4RunManagerKernel::GetRunManagerKernel()->GetTrackingManager()->SetVerboseLevel(0);
  fManager->SetVerboseLevel(1);
  mFileName = filename;
  mNumberOfInit++;
}
//==================================================
