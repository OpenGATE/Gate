/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateGenericWrapperProcess.hh"

#include "G4Track.hh"
#include "G4VParticleChange.hh"
#include <assert.h>
#include <vector>

#include "G4ParticleChangeForGamma.hh"
#include "G4ParticleChangeForLoss.hh"
#include "GateVProcess.hh"

#include "G4MaterialTable.hh"
//#include "G4MaterialCutsCouple.hh"
#include "G4ParticleTable.hh"

GenericWrapperProcess::GenericWrapperProcess(G4String name )
{
  mSplitFactor = 1;
  mNSecondaries = 0;
  mActive = true;
  mWeight=1.;
  mNFilterPrimary=0;
  mNFilterSecondary=0;
  pFilterManagerSecondary=0;
  pFilterManagerPrimary=0;
  SetGenericWrapperProcess( name);
  mCSEFactor=1.;
  mCSEnhancement = false;
  mSplitting = false;
  mkeepSec  = false;
  mInitCS = false;
  mRR = false;

  mNbins = 10000; // 10000 bins between 0 and 50 MeV
  mEneMax = 50.;  // 10000 bins between 0 and 50 MeV

  emcalc = new G4EmCalculator;
}

GenericWrapperProcess::~GenericWrapperProcess() 
{
  if(pFilterManagerPrimary) delete pFilterManagerPrimary;
  if(pFilterManagerSecondary) delete pFilterManagerSecondary;
}


void GenericWrapperProcess::SetGenericWrapperProcess(G4String name)
{
  pFilterManagerPrimary = new GateFilterManager(name+"/prim");
  pFilterManagerSecondary = new GateFilterManager(name+"/second");
}

void GenericWrapperProcess::Initialisation(G4String partName)
{
  double sig1 = 0.;
  double sig2 = 0.;

  double energy = 0.;

  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* partDef = particleTable->FindParticle(partName);
  G4ProcessVector * pList = partDef->GetProcessManager()-> GetProcessList();
  const G4MaterialTable* matTbl = G4Material::GetMaterialTable();

  G4String material;
 
  for(size_t k=0;k<G4Material::GetNumberOfMaterials();k++)
    {
      material = (*matTbl)[k]->GetName();

      for(int j= 0 ; j<mNbins;j++){// build table of cross section ratios in function of energy -> mNbins  bins between 0 and mEneMax MeV
	energy = mEneMax/mNbins*j;
	if(energy==0.) energy = 0.0000001;
	sig1 = 0.;
	sig2 = 0.;
	for(unsigned int i =0;i<pList->size();i++){
	  G4String proName = (*pList)[i]->GetProcessName();

	  G4String realName= proName;
	  if(proName.find("Wrapped")<10000) realName = realName.replace(0,7,"");
	  if(proName==GetProcessName()) 
	    sig1 = emcalc->ComputeCrossSectionPerVolume(energy, partName , realName, material, 0.0000001);
	  else {
	    sig2 += emcalc->ComputeCrossSectionPerVolume(energy, partName , realName, material, 0.0000001);
	  }
	}
  
	if(sig1!=0 || sig2!=0) theListOfBranchRatioFactor[material].push_back( (sig2 + mCSEFactor*sig1)/(mCSEFactor*(sig1+sig2)) );
	else theListOfBranchRatioFactor[material].push_back(0.);
      }
    }

  mInitCS = true;
}

G4double GenericWrapperProcess::PostStepGetPhysicalInteractionLength(const G4Track& track, G4double previousStepSize, G4ForceCondition* condition) 
{
  G4double interL =0.;

  if( mCSEnhancement && pFilterManagerPrimary->Accept(&track) )
    {
      previousStepSize=0.;
      interL = pRegProcess->PostStepGetPhysicalInteractionLength(track, previousStepSize, condition )/mCSEFactor;
    }
  else {
    interL = pRegProcess->PostStepGetPhysicalInteractionLength(track, previousStepSize, condition );
  }

  return  interL; 
}



G4VParticleChange* 
GenericWrapperProcess::PostStepDoIt(const G4Track& track, const G4Step& step)
{
 
  G4VParticleChange* particleChange(0);
  
  if (!mActive || (!mCSEnhancement && !mSplitting) || !pFilterManagerPrimary->Accept(&step) ) {//?
    particleChange = pRegProcess->PostStepDoIt(track, step);
    assert (0 != particleChange);//?

    mNSecondaries += particleChange->GetNumberOfSecondaries();//?

    return particleChange;
  }

  G4double  kine     = step.GetPostStepPoint()-> GetKineticEnergy();
  G4ThreeVector dir  = step.GetPostStepPoint()-> GetMomentumDirection();

  G4int i(0);
  //G4double weight = track.GetWeight()*mWeight/mCSEFactor;
  G4double weight = track.GetWeight()*mWeight;
  G4double rdm = 0.;
  // Secondary store
  std::vector<G4Track*> secondaries;
  std::vector<G4Track*> filteredSecondaries;
  bool validFinalState = false;

  // Loop over PostStepDoIt method to generate multiple secondaries.
  for (i=0; i<mSplitFactor; i++) {    
    validFinalState = false;
    particleChange = pRegProcess->PostStepDoIt(track, step);
    particleChange->SetVerboseLevel(0);//?

    G4int j(0);

    if(mSplitting && mRR)
      rdm = G4RandFlat::shoot(mWeight);

    if( (rdm<1. && weight>1.) || weight<1. || mCSEnhancement) {// =?
      if(pFilterManagerSecondary->GetNumberOfFilters()>0){
        for (j=0; j<particleChange->GetNumberOfSecondaries(); j++) {
          if(pFilterManagerSecondary->Accept(particleChange->GetSecondary(j)) ) validFinalState = true;
        }
        if(validFinalState)
          for (j=0; j<particleChange->GetNumberOfSecondaries(); j++) {
            ///particleChange->GetSecondary(j)->SetWeight(weight);
	    secondaries.push_back(new G4Track(*(particleChange->GetSecondary(j))));
	  }
	else if(mkeepSec){
	  rdm =  G4RandFlat::shoot(mWeight);
          if(rdm<1.) {
	    for (j=0; j<particleChange->GetNumberOfSecondaries(); j++) {
	      //filteredSecondaries.push_back(((particleChange->GetSecondary(j))));
              //particleChange->GetSecondary(j)->SetWeight(1.);
	      filteredSecondaries.push_back(new G4Track(*(particleChange->GetSecondary(j))));
	    }
	  }
	}
      }//if(pFilterManagerSecondary->GetNumberOfFilters()>0)
      else {
	for (j=0; j<particleChange->GetNumberOfSecondaries(); j++) {
	  secondaries.push_back(new G4Track(*(particleChange->GetSecondary(j))));
	}
      }//else
    } 
    // particleChange->SetNumberOfSecondaries(secondaries.size()+filteredSecondaries.size());
    //particleChange->Clear();
  }//for mSplitFactor



  if(mCSEnhancement){

    G4ParticleDefinition *partDef = track.GetDefinition();
    G4String partName =  partDef->GetParticleName();
    //G4ProcessManager * pMan = pDef->GetProcessManager() ;
    /*G4ProcessVector * pList = partDef->GetProcessManager()-> GetProcessList();

      for(int i =0;i<pList->size();i++){
      G4String proName = (*pList)[i]->GetProcessName();
      G4cout<<proName<<"  "<<GetProcessName()<< Gateendl;
      if(proName==GetProcessName()) 
      sig1 = emcalc->ComputeCrossSectionPerVolume(track.GetKineticEnergy(), partName , proName, track.GetMaterial()->GetName(), 0.0000001);
      else 
      sig2 += emcalc->ComputeCrossSectionPerVolume(track.GetKineticEnergy(), partName , proName, track.GetMaterial()->GetName(), 0.0000001);
      }
      double frac = 0.;
      G4cout<<sig1<<"  "<<sig2<< Gateendl;
      if(sig1!=0 || sig2!=0) frac = (sig2 + mCSEFactor*sig1)/(mCSEFactor*(sig1+sig2));
      else frac =0.;
    */
 
    double frac = 1./mCSEFactor;
    /*    if(partName=="gamma"){
          if(!mInitCS) Initialisation(partName);

          int bin = int(floor(track.GetKineticEnergy()/(50./10000.)));
          G4String material = track.GetMaterial()->GetName();

          double y2 = theListOfBranchRatioFactor[material][bin+1];
          double y1 = theListOfBranchRatioFactor[material][bin];

          frac = y2 + ( (mEneMax*(bin+1))/mNbins - track.GetKineticEnergy())/(mEneMax/mNbins)*(y1 - y2);
          G4cout<<"frac = "<<frac<<" y1= "<<y1<<" y2= "<<y2<< Gateendl;

          G4ProcessManager * pMan = partDef->GetProcessManager() ;
          G4ProcessVector * pList = partDef->GetProcessManager()-> GetProcessList();
          double sig1=0.;
          double sig2=0.;
          for(int i =0;i<pList->size();i++){
          G4String proName = (*pList)[i]->GetProcessName();
          // G4cout<<proName<<"  "<<GetProcessName()<< Gateendl;
          G4String realName= proName;
	  if(proName.find("Wrapped")<10000) realName = realName.replace(0,7,"");

	  if(proName==GetProcessName()) {
	  sig1 = emcalc->ComputeCrossSectionPerVolume(track.GetKineticEnergy(), partName ,realName , track.GetMaterial()->GetName(), 0.000000);
	  G4cout<<"proName  "<<proName<<"  "<<sig1<<"  "<<track.GetKineticEnergy()<<"  "<<realName<<"  "<< emcalc->ComputeMeanFreePath(track.GetKineticEnergy(), partName ,realName , track.GetMaterial()->GetName(), 0.000000)  << Gateendl;
	  }
          else 
	  G4cout<<"proName  "<<proName<<"  "<<emcalc->ComputeCrossSectionPerVolume(track.GetKineticEnergy(), partName , proName, track.GetMaterial()->GetName(), 0.000000 )<<"  "<<  emcalc->ComputeMeanFreePath(track.GetKineticEnergy(), partName , proName, track.GetMaterial()->GetName(), 0.000000) << Gateendl;
	  sig2 += emcalc->ComputeCrossSectionPerVolume(track.GetKineticEnergy(), partName , proName, track.GetMaterial()->GetName(), 0.000000);
          }
          frac = 0.;
          //G4cout<<sig1<<"  "<<sig2<< Gateendl;
          if(sig1!=0 || sig2!=0) frac = (sig2 + mCSEFactor*sig1)/(mCSEFactor*(sig1+sig2));
          //if(sig1!=0 || sig2!=0) frac = (mCSEFactor*sig1)/(sig2+mCSEFactor*sig1);
          else frac =0.;
          G4cout<<"frac2 = "<<frac<<"  "<<sig1<<"  "<< sig2 << Gateendl;


          }*/

    weight /= mCSEFactor;
    
    frac = 1./mCSEFactor;
    rdm = G4RandFlat::shoot();
    if(rdm<(1.-frac)){
      if(partName=="gamma"){
        dynamic_cast<G4ParticleChangeForGamma*>(particleChange)->ProposeMomentumDirection(dir);
        dynamic_cast<G4ParticleChangeForGamma*>(particleChange)->SetProposedKineticEnergy(kine);
        dynamic_cast<G4ParticleChangeForGamma*>(particleChange)->ProposeTrackStatus(fAlive); 
      }
      else{
        dynamic_cast<G4ParticleChangeForLoss*>(particleChange)->SetProposedMomentumDirection(dir);
        dynamic_cast<G4ParticleChangeForLoss*>(particleChange)->SetProposedKineticEnergy(kine);
        dynamic_cast<G4ParticleChangeForLoss*>(particleChange)->ProposeTrackStatus(fAlive);
      }
      //      particleChange->ProposeMomentumDirection(dir);
      //      particleChange->ProposeEnergy(kine);
    }
  }




  // Configure particleChange to handle multiple secondaries. Other 
  // data is unchanged
  if(mWeight!=1. || mCSEFactor!=1.){
    //particleChange->Clear();
    particleChange->SetNumberOfSecondaries(secondaries.size()+filteredSecondaries.size());
    particleChange->SetSecondaryWeightByProcess(true);

    // Add all secondaries 
    std::vector<G4Track*>::iterator iter2 = secondaries.begin();

    while (iter2 != secondaries.end()) {
      G4Track* myTrack = *iter2;
      myTrack->SetWeight(weight);

      // particleChange takes ownership
      particleChange->AddSecondary(myTrack); 
    
      iter2++;
    }

    std::vector<G4Track*>::iterator iter = filteredSecondaries.begin();

    while (iter != filteredSecondaries.end()) {
      G4Track* myTrack = *iter;
      myTrack->SetWeight(1.);

      // particleChange takes ownership
      particleChange->AddSecondary(myTrack); 
    
      iter++;
    }

    mNSecondaries += secondaries.size()+filteredSecondaries.size();//?
    particleChange->SetSecondaryWeightByProcess(false); //must be set to false else track keep the true state
  }

  return particleChange;
}

void GenericWrapperProcess::SetSplitFactor(G4double n) 
{
  if(n>=1){
    //Splitting
    mSplitFactor=int(n);
    mWeight=1./mSplitFactor;//n;
    mSplitting = true;
  }
  else if(n<=0) mSplitFactor=0;
  else{
    //Russian Roulette
    mSplitFactor = 1;
    mWeight=1./n;
    mSplitting = true;
    mRR = true;
  }
}

void GenericWrapperProcess::SetCSEFactor(G4double n) 
{
  mCSEFactor=n;
  mCSEnhancement = true;
}

void GenericWrapperProcess::SetIsActive(G4bool active) 
{
  mActive = active;
}

G4bool GenericWrapperProcess::GetIsActive() 
{
  return mActive;
}

G4int GenericWrapperProcess::GetFactor() 
{
  return mSplitFactor;
}

G4int GenericWrapperProcess::GetNSecondaries() 
{
  return mNSecondaries;
}
 
/*void GenericWrapperProcess::SetProcessManager(const G4ProcessManager* procMan)
  {
  //G4VProcess::SetProcessManager(procMan);
  G4cout<<" set process manager : "<<procMan<< Gateendl;
  // proc = new G4ProcessManager(0);
  pRegProcess->SetProcessManager(procMan);
  //proc->add(pRegProcess);
  //proc->GetProcessList()->insert(pRegProcess);
  //proc->GetProcessList()->insert(pRegProcess);
  //proc->SetProcessActivation(pRegProcess, true);
  aProcessManager = procMan; 
  }

  const G4ProcessManager* GenericWrapperProcess::GetProcessManager()
  {
  //return G4VProcess::GetProcessManager();
  G4cout<<" get process manager : "<<aProcessManager<< Gateendl;
  G4cout<<" get process manager : "<<aProcessManager<< Gateendl;
  return aProcessManager;
  }*/
