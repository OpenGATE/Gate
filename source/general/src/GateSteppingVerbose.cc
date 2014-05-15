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
GateSteppingVerbose::GateSteppingVerbose():G4SteppingVerbose()
{
  pStepTime = new G4SliceTimer;
  pTrackTime = new G4SliceTimer;
  pStepTime->Clear();

  mTempoEnergy = 0.;
  mTempoProcess = "";
  mTempoParticle = "";
  mTempoVolume = "";
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
  //theListOfStep.clear();

  theListOfVolume.clear(); 
  theListOfProcess.clear(); 
  theListOfParticle.clear(); 
  theListOfTime.clear();

  theListOfTrack.clear();

  theListOfVolumeAtTrackLevel.clear();
  theListOfProcessAtTrackLevel.clear();
  theListOfParticleAtTrackLevel.clear();
  theListOfTimerAtTrackLevel.clear();
  theListOfEnergyAtTrackLevel.clear();
}
//==================================================


//==================================================
void GateSteppingVerbose::NewStep()
{
  if(!mIsTrackingStep)G4SteppingVerbose::NewStep();
  if(!mIsTrackingStep) return

  CopyState();

  if(mTempoVolume != fTrack->GetVolume()->GetName())
  { 
    theListOfTimerAtTrackLevel[currentTrack]->Stop();
    mTempoVolume = fTrack->GetVolume()->GetName();

    bool knownState = false;

    for(unsigned int i = 0; i<theListOfParticleAtTrackLevel.size();i++)
    {
      if(theListOfParticleAtTrackLevel[i] == mTempoParticle &&
      theListOfProcessAtTrackLevel[i] == mTempoProcess &&
      theListOfVolumeAtTrackLevel[i] == mTempoVolume &&
      theListOfEnergyAtTrackLevel[i] == GetEnergyRange(mTempoEnergy))
      {
        if(knownState) GateError("This state was already found.");
        theListOfTimerAtTrackLevel[i]->Start();// += pTrackTime->GetUserElapsed();
        currentTrack = i;
        knownState=true;
      }
    }

    if(!knownState)
    {
      theListOfProcessAtTrackLevel.push_back(mTempoProcess);
      theListOfParticleAtTrackLevel.push_back(mTempoParticle); 
      theListOfVolumeAtTrackLevel.push_back(mTempoVolume);
      theListOfEnergyAtTrackLevel.push_back(GetEnergyRange(mTempoEnergy));
      theListOfTimerAtTrackLevel.push_back(new G4SliceTimer);
      currentTrack = theListOfTimerAtTrackLevel.size()-1;
      theListOfTimerAtTrackLevel[currentTrack]->Start();
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

  G4String particleName = fStep->GetTrack()->GetDefinition()->GetParticleName();
  G4String volumeName = currentVol->GetName();
  G4String processName = "NoProcess";
  //if(fStep->GetPostStepPoint()->GetProcessDefinedStep()) processName = fStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  if(fCurrentProcess) processName = fCurrentProcess->GetProcessName();
 //G4cout<<particleName<<"   "<<processName<<G4endl;
  //G4double energy = fStep->GetPreStepPoint()->GetKineticEnergy();
  G4int energyRange = GetEnergyRange(fStep->GetPreStepPoint()->GetKineticEnergy());
  bool knownState = false;

  for(unsigned int i = 0; i<theListOfParticle.size();i++)
  {
      if(theListOfParticle[i] == particleName && theListOfProcess[i] ==  processName && theListOfVolume[i] == volumeName && theListOfEnergy[i] == energyRange)
      {
        if(knownState) GateError("This state was already found.");
	theListOfTimer[i] += pTempTime->GetUserElapsed();
        //theListOfTimer[i]->Start();// += pStepTime->GetUserElapsed();
	knownState=true;
      }
  }


  if(!knownState)
  {
      //G4cout<<mTempoParticle<<"  "<<mTempoProcess<<"  "<<theListOfProcess.size()<<"  "<<theListOfParticle.size()<<G4endl;
      theListOfProcess.push_back(processName);
      theListOfParticle.push_back(particleName); 
      theListOfTimer.push_back(pTempTime->GetUserElapsed());
      //theListOfTimer.push_back(new G4SliceTimer());
      //theListOfTimer[theListOfTimer.size()-1]->Clear();
      //theListOfTimer[theListOfTimer.size()-1]->Start();
      theListOfVolume.push_back(volumeName);
      theListOfEnergy.push_back(energyRange);
  }


/*  GateInfoForSteppingVerbose * infoStep = new GateInfoForSteppingVerbose();
  infoStep->SetEnergy(energy);
  infoStep->SetVolume(volumeName);
  infoStep->SetProcess(processName);
  infoStep->SetParticle(particleName);
  infoStep->SetTime(pStepTime->GetUserElapsed());
  theListOfStep.push_back(infoStep);*/
//G4cout<<"Time = "<<pStepTime->IsValid()<<G4endl;
//G4cout<<"Time1 = "<<pStepTime->GetUserElapsed()<<G4endl;
//G4cout<<"Time2 = "<<pStepTime->GetRealElapsed()<<G4endl;
//  bool knownState = false;

  //G4cout<<theListOfStep.size()<<G4endl;



}
//==================================================


//==================================================
void GateSteppingVerbose::TrackingStarted()
{
  if(!mIsTrackingStep)G4SteppingVerbose::TrackingStarted();
  CopyState();

  mTempoEnergy   = fTrack->GetKineticEnergy();
  if( fTrack->GetCreatorProcess()) mTempoProcess  = fTrack->GetCreatorProcess()->GetProcessName();
  else mTempoProcess  = "Source";
  mTempoParticle = fTrack->GetDefinition()->GetParticleName();
  mTempoVolume   = fTrack->GetVolume()->GetName();//GetLogicalVolumeAtVertex()->GetName();

  bool knownState = false;

  for(unsigned int i = 0; i<theListOfParticleAtTrackLevel.size();i++)
  {
    if(theListOfParticleAtTrackLevel[i] == mTempoParticle &&
    theListOfProcessAtTrackLevel[i] == mTempoProcess &&
    theListOfVolumeAtTrackLevel[i] == mTempoVolume &&
    theListOfEnergyAtTrackLevel[i] == GetEnergyRange(mTempoEnergy))
    {
      if(knownState) GateError("This state was already found.");
      theListOfTimerAtTrackLevel[i]->Start();// += pTrackTime->GetUserElapsed();
      currentTrack = i;
      knownState=true;
    }
  }

  if(!knownState)
  {
    theListOfProcessAtTrackLevel.push_back(mTempoProcess);
    theListOfParticleAtTrackLevel.push_back(mTempoParticle); 
    theListOfVolumeAtTrackLevel.push_back(mTempoVolume);
    theListOfEnergyAtTrackLevel.push_back(GetEnergyRange(mTempoEnergy));
    theListOfTimerAtTrackLevel.push_back(new G4SliceTimer);
    currentTrack = theListOfTimerAtTrackLevel.size()-1;
    theListOfTimerAtTrackLevel[currentTrack]->Start();
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
  theListOfTimerAtTrackLevel[currentTrack]->Stop();

  /*GateInfoForSteppingVerbose * info = new GateInfoForSteppingVerbose();
  info->SetEnergy(mTempoEnergy);
  info->SetVolume(mTempoVolume);
  info->SetProcess(mTempoProcess);
  info->SetParticle(mTempoParticle);
  info->SetTime(pTrackTime->GetUserElapsed());
  theListOfTrack.push_back(info);*/

  


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
    G4cout<<theListOfParticle[i]<<"     "<<theListOfTimer[i]->GetUserElapsed()<<"   "<<G4BestUnit(theListOfTimer[i]->GetRealElapsed() , "Time")<<G4endl;
//G4BestUnit(theListOfUsedTime[i] , "Time");
  }*/

  std::vector<G4String> theListOfUsedParticle; 
  std::vector<G4double> theListOfUsedTime;

//  std::vector<G4String> theListOfUsedVolume; 
//  std::vector<G4double> theListOfUsedVolumeTime; 
//  std::vector<G4String> theListOfUsedProcess; 
//  std::vector<G4double> theListOfUsedProcessTime; 
//  std::vector<G4int> theListOfUsedEnergy;


  bool alreadyUsed = false;

  for(unsigned int i = 0; i<theListOfParticleAtTrackLevel.size();i++)
  {
     alreadyUsed = false;
     for(unsigned int j = 0; j<theListOfUsedParticle.size();j++)
     {
        if(theListOfParticleAtTrackLevel[i]==theListOfUsedParticle[j]) 
        {
          theListOfUsedTime[j] +=  theListOfTimerAtTrackLevel[i]->GetUserElapsed();
          alreadyUsed = true;
        }
     }
     if(!alreadyUsed) 
     {
       theListOfUsedParticle.push_back(theListOfParticleAtTrackLevel[i]);
       theListOfUsedTime.push_back(theListOfTimerAtTrackLevel[i]->GetUserElapsed());
     }
  }

  std::ofstream os;
  OpenFileOutput(mFileName, os);
  os<<G4endl;
   if(mIsTrackingStep) os<<" Total time elapsed during physical part of steps = "<<pStepTime->GetUserElapsed()<<" s"<<G4endl;

  os<<G4endl;
  os<<G4endl;
  os<<"----> Track level"<<G4endl;
  os<<G4endl;
  for(unsigned int i = 0; i<theListOfUsedParticle.size();i++)
  {
    os<<"------------------------------------------------------------------------"<<G4endl;
    os<<theListOfUsedParticle[i]<<"     "<<theListOfUsedTime[i]<<" s"<<G4endl;
    DisplayTrack(theListOfUsedParticle[i], os);
    os<<G4endl;
    os<<"------------------------------------------------------------------------"<<G4endl;
    os<<G4endl;
    os<<G4endl;
 }

  alreadyUsed = false;

  theListOfUsedParticle.clear(); 
  theListOfUsedTime.clear();

  for(unsigned int i = 0; i<theListOfParticle.size();i++)
  {
     //os<<theListOfParticle[i]<<"  "<<theListOfProcess[i]<<G4endl;
     alreadyUsed = false;
     for(unsigned int j = 0; j<theListOfUsedParticle.size();j++)
     {
        if(theListOfParticle[i]==theListOfUsedParticle[j]) 
        {
          theListOfUsedTime[j] +=  theListOfTimer[i];
          //theListOfUsedTime[j] +=  theListOfTimer[i]->GetUserElapsed();
          alreadyUsed = true;
        }
     }
     if(!alreadyUsed) 
     {
       theListOfUsedParticle.push_back(theListOfParticle[i]);
       theListOfUsedTime.push_back(theListOfTimer[i]);
       //theListOfUsedTime.push_back(theListOfTimer[i]->GetUserElapsed());
     }
  }

 
  os<<G4endl;
  if(mIsTrackingStep) {
    os<<G4endl;
    os<<"----> Physical step level"<<G4endl;
    os<<G4endl;

    for(unsigned int i = 0; i<theListOfUsedParticle.size();i++)
    {
      os<<"------------------------------------------------------------------------"<<G4endl;
      os<<theListOfUsedParticle[i]<<"     "<<theListOfUsedTime[i]<<" s"<<G4endl;
      DisplayStep(theListOfUsedParticle[i], os);
      os<<G4endl;
      os<<"------------------------------------------------------------------------"<<G4endl;
      os<<G4endl;
      os<<G4endl;


      //G4BestUnit(theListOfUsedTime[i] , "Time");
    }
  }
  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mFileName << G4endl);
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

  for(unsigned int i = 0; i<theListOfEnergyAtTrackLevel.size();i++)
  {
    if(theListOfEnergyAtTrackLevel[i]>energyMax) energyMax = theListOfEnergyAtTrackLevel[i];
  }


//Volume
  for(unsigned int i = 0; i<theListOfVolumeAtTrackLevel.size();i++)
  {
    theListOfUsedVolume[theListOfVolumeAtTrackLevel[i]] = 0.;
    theListOfTotalPerVolume[theListOfVolumeAtTrackLevel[i]] = 0.;
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; ++it)
  {
    os<<"\t"<<it->first;
  }
  os<<G4endl;
 

  for(int j = 3; j<=energyMax;j++)
  {
    for(unsigned int i = 0; i<theListOfVolumeAtTrackLevel.size();i++)
    {
      theListOfUsedVolume[theListOfVolumeAtTrackLevel[i]] = 0.;
    }  
    for(unsigned int i = 0; i<theListOfParticleAtTrackLevel.size();i++)
    {
      if(theListOfParticleAtTrackLevel[i] == particle)
      {
        if(j==theListOfEnergyAtTrackLevel[i])
        {
          theListOfUsedVolume[theListOfVolumeAtTrackLevel[i]] += theListOfTimerAtTrackLevel[i]->GetUserElapsed();
          theListOfTotalPerVolume[theListOfVolumeAtTrackLevel[i]] += theListOfTimerAtTrackLevel[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; ++it)
    {
      os<<"\t"<<it->second;
    }
    os<<G4endl;
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerVolume.begin() ; it!=theListOfTotalPerVolume.end() ; ++it)
  {
    os<<"\t"<<it->second;
  }
    os<<G4endl;


    os<<G4endl;    

//Process
  for(unsigned int i = 0; i<theListOfProcessAtTrackLevel.size();i++)
  {
    if(theListOfParticleAtTrackLevel[i] == particle) {
      theListOfUsedProcess[theListOfProcessAtTrackLevel[i]] = 0.;
      theListOfTotalPerProcess[theListOfProcessAtTrackLevel[i]] = 0.;
    }
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; ++it)
  {
    os<<"\t"<<it->first;
  }
  os<<G4endl;    
 

  for(int j = 3; j<=energyMax;j++)
  {
    for(unsigned int i = 0; i<theListOfProcessAtTrackLevel.size();i++)
    {
      if(theListOfParticleAtTrackLevel[i] == particle) theListOfUsedProcess[theListOfProcessAtTrackLevel[i]] = 0.;
    }  
    for(unsigned int i = 0; i<theListOfParticleAtTrackLevel.size();i++)
    {
      if(theListOfParticleAtTrackLevel[i] == particle)
      {
        if(j==theListOfEnergyAtTrackLevel[i])
        {
          theListOfUsedProcess[theListOfProcessAtTrackLevel[i]] += theListOfTimerAtTrackLevel[i]->GetUserElapsed();
	  theListOfTotalPerProcess[theListOfProcessAtTrackLevel[i]] += theListOfTimerAtTrackLevel[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; ++it)
    {
      os<<"\t"<<it->second;
    }
    os<<G4endl;   
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerProcess.begin() ; it!=theListOfTotalPerProcess.end() ; ++it)
  {
    os<<"\t"<<it->second;
  }
    os<<G4endl;


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

  for(unsigned int i = 0; i<theListOfEnergy.size();i++)
  {
    if(theListOfEnergy[i]>energyMax) energyMax = theListOfEnergy[i];
  }


//Volume
  for(unsigned int i = 0; i<theListOfVolume.size();i++)
  {
    theListOfUsedVolume[theListOfVolume[i]] = 0.;
    theListOfTotalPerVolume[theListOfVolume[i]] = 0.;
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; ++it)
  {
    os<<"\t"<<it->first;
  }
  os<<G4endl;
 

  for(int j = 3; j<=energyMax;j++)
  {
    for(unsigned int i = 0; i<theListOfVolume.size();i++)
    {
      theListOfUsedVolume[theListOfVolume[i]] = 0.;
    }  
    for(unsigned int i = 0; i<theListOfParticle.size();i++)
    {
      if(theListOfParticle[i] == particle)
      {
        if(j==theListOfEnergy[i])
        {
          theListOfUsedVolume[theListOfVolume[i]] += theListOfTimer[i];
          theListOfTotalPerVolume[theListOfVolume[i]] += theListOfTimer[i];
          //theListOfUsedVolume[theListOfVolume[i]] += theListOfTimer[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedVolume.begin() ; it!=theListOfUsedVolume.end() ; ++it)
    {
      os<<"\t"<<it->second;
    }
    os<<G4endl;    
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerVolume.begin() ; it!=theListOfTotalPerVolume.end() ; ++it)
  {
    os<<"\t"<<it->second;
  }
    os<<G4endl;

    os<<G4endl;    

//Process
  for(unsigned int i = 0; i<theListOfProcess.size();i++)
  {
   // os<<theListOfParticle[i]<<"  "<<theListOfProcess[i]<<G4endl;
    if(theListOfParticle[i] == particle) {
      theListOfUsedProcess[theListOfProcess[i]] = 0.;
      theListOfTotalPerProcess[theListOfProcess[i]] = 0.;
    }
  }

  os<<"  ";
  for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; ++it)
  {
    os<<"\t"<<it->first;
  }
  os<<G4endl;    
 

  for(int j = 3; j<=energyMax;j++)
  {
    for(unsigned int i = 0; i<theListOfProcess.size();i++)
    {
      if(theListOfParticle[i] == particle) theListOfUsedProcess[theListOfProcess[i]] = 0.;
    }  
    for(unsigned int i = 0; i<theListOfParticle.size();i++)
    {
      if(theListOfParticle[i] == particle)
      {
        if(j==theListOfEnergy[i])
        {
          theListOfUsedProcess[theListOfProcess[i]] += theListOfTimer[i];
	  theListOfTotalPerProcess[theListOfProcess[i]]+= theListOfTimer[i];
          //theListOfUsedProcess[theListOfProcess[i]] += theListOfTimer[i]->GetUserElapsed();
        }
      }
    }
    if(j==3) os<<"0 - 1 keV";
    if(j>3) os<<G4BestUnit(pow(10,j-3),"Energy")<<" - "<<G4BestUnit(pow(10,j-2),"Energy");
    for(std::map<G4String,G4double>::iterator it=theListOfUsedProcess.begin() ; it!=theListOfUsedProcess.end() ; ++it)
    {
      os<<"\t"<<it->second;
    }
    os<<G4endl;   
  }

  os<<"Total";
  for(std::map<G4String,G4double>::iterator it=theListOfTotalPerProcess.begin() ; it!=theListOfTotalPerProcess.end() ; ++it)
  {
    os<<"\t"<<it->second;
  }
    os<<G4endl;

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
