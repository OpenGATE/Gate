/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#include "G4Track.hh"
#include "G4VParticleChange.hh"
#include <assert.h>
#include <vector>

#include "GateVProcess.hh"
#include "G4ComptonScattering.hh"
#include "G4Hybridino.hh"
#include "G4HybridComptonProcess.hh"

G4HybridComptonProcess::G4HybridComptonProcess()
{
  mInVolumeSplit = 150;
  mOutVolumeSplit = 150;
  mCurrentSplit = 150;
}

G4HybridComptonProcess::~G4HybridComptonProcess() 
{
}

G4VEmProcess* G4HybridComptonProcess::GetEmProcess()
{
  return dynamic_cast<G4VEmProcess*>(pRegProcess);
}

G4VParticleChange *G4HybridComptonProcess::PostStepDoIt(const G4Track& track, const G4Step& step)
{
  G4VParticleChange* particleChange(0);
  G4Step * myStep = new G4Step;
  G4Step* newStep;
  myStep->SetTrack(step.GetTrack());
  myStep->SetPreStepPoint(step.GetPreStepPoint());
  myStep->SetPostStepPoint(step.GetPostStepPoint());
  myStep->SetStepLength(step.GetStepLength());

  std::vector<G4DynamicParticle*> secondaries;
  secondaries.reserve(mCurrentSplit);
  for(int i = 0; i < mCurrentSplit; i++)
  {
    particleChange = pRegProcess->PostStepDoIt(track, step);
    particleChange->SetVerboseLevel(0);
    newStep = particleChange->UpdateStepForPostStep(myStep);
        
    G4double Energy = newStep->GetPostStepPoint()->GetKineticEnergy();
    G4ThreeVector mmt = newStep->GetPostStepPoint()->GetMomentumDirection();
    G4ParticleDefinition *hybridino = G4Hybridino::Hybridino();
    G4DynamicParticle* newgamma = new G4DynamicParticle(hybridino, mmt, Energy);
    secondaries.push_back(newgamma);
  }
  
  particleChange->SetNumberOfSecondaries(mCurrentSplit);

  std::vector<G4DynamicParticle*>::iterator iter = secondaries.begin();
  G4double weight = track.GetWeight()/mCurrentSplit;
  while (iter != secondaries.end())
  {
    G4DynamicParticle* particle = *iter;
    G4Track *myTrack = new G4Track(particle, track.GetGlobalTime(), track.GetPosition());
    myTrack->SetWeight(weight);
    particleChange->AddSecondary(myTrack); 
    iter++;
  }

  return particleChange;
}

