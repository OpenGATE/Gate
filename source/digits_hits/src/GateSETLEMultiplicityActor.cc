/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class GateSETLEMultiplicityActor : 
  \brief 
*/

#ifndef GATESETLEMULTIPLICITYACTOR_CC
#define GATESETLEMULTIPLICITYACTOR_CC

#include "GateSETLEMultiplicityActor.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"
#include "GateActorManager.hh"
#include "GateDetectorConstruction.hh"
#include "GateMultiSensitiveDetector.hh"
#include "G4Event.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateSETLEMultiplicityActor::GateSETLEMultiplicityActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateSETLEMultiplicityActor() -- begin\n");
  mMaterialHandler = GateMaterialMuHandler::GetInstance();
  GateDebugMessageDec("Actor",4,"GateSETLEMultiplicityActor() -- end\n");

  mIsHybridinoEnabled = false;
  mDefaultPrimaryMultiplicity = 0;
  mDefaultSecondaryMultiplicity = 0;
  mSecondaryMultiplicityMap.clear();
  mProcessListForGamma = 0;
  mHybridino = G4Hybridino::Hybridino();
  
  singleton_SETLEMultiplicityActor = this;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor 
GateSETLEMultiplicityActor::~GateSETLEMultiplicityActor() {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateSETLEMultiplicityActor::Construct()
{
  GateMessage("Actor", 0, " SETLEMultiplicityActor auto-construction\n");
  
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
    
  if((mDefaultPrimaryMultiplicity<0) || (mDefaultSecondaryMultiplicity<0)) {
    GateError("Multiplicity cannot be inferior to 0 (Mprim = " << mDefaultPrimaryMultiplicity << ", Msec = " << mDefaultSecondaryMultiplicity << ")");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEMultiplicityActor::SetMultiplicity(bool b, int mP, int mS, G4VPhysicalVolume *v) 
{
  // set hybridino flag
  mIsHybridinoEnabled = b;
  
  // keep the highest multiplicity as default value
  if(mP > mDefaultPrimaryMultiplicity) { mDefaultPrimaryMultiplicity = mP; }
  if(mS > mDefaultSecondaryMultiplicity) { mDefaultSecondaryMultiplicity = mS; }

  // register expTLEDoseActor's volume
  std::map<G4VPhysicalVolume *,int>::iterator it = mSecondaryMultiplicityMap.find(v);  
  if(it == mSecondaryMultiplicityMap.end()) { mSecondaryMultiplicityMap.insert(std::make_pair(v,mS)); }
  else { GateError("Number of 'hybridDoseActor' attached to '" << v->GetName() << "' is too large (1 maximum)"); }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Event
void GateSETLEMultiplicityActor::BeginOfEventAction(const G4Event *event)
{
  if(!mProcessListForGamma) { mProcessListForGamma = G4Gamma::Gamma()->GetProcessManager()->GetProcessList(); }
  mListOfHybridTrack.clear();
  mListOfHybridWeight.clear();
  mListOfRaycasting.clear();

  GateVSource* source = GateSourceMgr::GetInstance()->GetSource(GateSourceMgr::GetInstance()->GetCurrentSourceID());
  if(source->GetParticleDefinition()->GetParticleName() == "gamma")
    {
      if(mIsHybridinoEnabled)
        {
          G4Event *modifiedEvent = const_cast<G4Event *>(event);
          int vertexNumber = event->GetNumberOfPrimaryVertex();

          for(int i=0; i<mDefaultPrimaryMultiplicity; i++)
            {
              vertexNumber += source->GeneratePrimaries(modifiedEvent);
              G4PrimaryParticle *hybridParticle = modifiedEvent->GetPrimaryVertex(vertexNumber-1)->GetPrimary();
              while(hybridParticle != 0)
                {
                  hybridParticle->SetParticleDefinition(mHybridino);
                  hybridParticle = hybridParticle->GetNext();
                }      
            }
        }
      else
        {
          G4Event *modifiedEvent = new G4Event();
          int vertexNumber = modifiedEvent->GetNumberOfPrimaryVertex();
          double weight = 1.0 / mDefaultPrimaryMultiplicity;
          for(int i=0; i<mDefaultPrimaryMultiplicity; i++)
            {
              vertexNumber += source->GeneratePrimaries(modifiedEvent);
              G4ThreeVector position = modifiedEvent->GetPrimaryVertex(vertexNumber-1)->GetPosition();
              G4PrimaryParticle *hybridParticle = modifiedEvent->GetPrimaryVertex(vertexNumber-1)->GetPrimary();
              while(hybridParticle != 0)
                {
                  // create a hybrid struct for raycasting
                  // primary or not - energy - weight - position - direction
                  mListOfRaycasting.push_back(RaycastingStruct(true, hybridParticle->GetKineticEnergy(), weight,
                                                               position, hybridParticle->GetMomentumDirection()));
                  hybridParticle = hybridParticle->GetNext();
	  
                }
            }
      
          delete modifiedEvent;
        }
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEMultiplicityActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  mCurrentTrackIndex = -1;
  mCurrentHybridTrackWeight = 1.;
  if(t->GetParticleDefinition()->GetParticleName() == "hybridino")
    {
      //     GateMessage("Actor", 0, "track = " << t << " parentID = " << t->GetParentID() << Gateendl);
      if(t->GetParentID() == 0)
        {
          mCurrentHybridTrackWeight = t->GetWeight() / mDefaultPrimaryMultiplicity;
        }
      else
        {
          for(unsigned int i=0; i<mListOfHybridTrack.size(); i++)
            {
              if(mListOfHybridTrack[i] == t)
                {
                  mCurrentTrackIndex = i;
                  mCurrentHybridTrackWeight = mListOfHybridWeight[i];
                  break;
                }
            }
          if(mCurrentTrackIndex == -1) { GateError("Could not find the following hybrid track : " << t); }
        }
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEMultiplicityActor::PostUserTrackingAction(const GateVVolume *, const G4Track *)
{
  if(mCurrentTrackIndex > -1)
    {
      mListOfHybridTrack.erase(mListOfHybridTrack.begin() + mCurrentTrackIndex);
      mListOfHybridWeight.erase(mListOfHybridWeight.begin() + mCurrentTrackIndex);
    }

  //   for(unsigned int i=0; i<mListOfHybridTrack.size(); i++) { GateMessage("Actor", 0, "track = " << mListOfHybridTrack[i] << " weight = " << mListOfHybridWeight[i] << Gateendl); }
  //   GateMessage("Actor", 0, " \n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateSETLEMultiplicityActor::UserSteppingAction(const GateVVolume *, const G4Step * step)
{
  G4String particleName = step->GetTrack()->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();
  if(particleName == "hybridino")
    {
      G4double stepLength = step->GetStepLength();
    
      // Apply exponential attenuation if stepLength > 0
      if(stepLength > 0.) 
        {
          const G4MaterialCutsCouple *couple = step->GetPreStepPoint()->GetMaterialCutsCouple();
          G4double energy = step->GetPreStepPoint()->GetKineticEnergy();
          G4double mu = mMaterialHandler->GetMu(couple, energy);    
          mCurrentHybridTrackWeight = mCurrentHybridTrackWeight * exp(-mu*stepLength/10.);
        }
      //     GateMessage("ActorMult", 0, "hybridWeight = " << currentHybridTrackWeight << Gateendl);
    }
  else if(particleName == "gamma")
    {
      G4String processName = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();    
      if(processName != "Transportation")
        {
          G4VProcess *currentProcess = 0;
          for(unsigned int i=0; i<mProcessListForGamma->size(); i++)
            {
              if((*mProcessListForGamma)[i]->GetProcessName() == processName)
                {
                  currentProcess = (*mProcessListForGamma)[i];
                  break;
                }      
            }
          if(currentProcess == 0) { GateError("the process doesn't exist"); }
      
          if(processName == "PhotoElectric" || processName == "phot")
            {
              G4TrackVector *trackVector = (const_cast<G4Step *>(step))->GetfSecondary();
              int trackVectorLength = trackVector->size();
              for(int t=0; t<trackVectorLength; t++)
                {
                  G4String secondaryName = (*trackVector)[t]->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

                  // Is it a fluorescence gamma ??
                  if(secondaryName == "gamma")
                    {
                      // Get the constant datas for this fluorescence gamma
                      int currentSecondaryMultiplicity;
                      std::map<G4VPhysicalVolume *,int>::iterator it = mSecondaryMultiplicityMap.find(step->GetTrack()->GetVolume());
                      if(it == mSecondaryMultiplicityMap.end()) { currentSecondaryMultiplicity = mDefaultSecondaryMultiplicity; }
                      else { currentSecondaryMultiplicity = it->second; }
	    
                      G4double energy = (*trackVector)[t]->GetKineticEnergy();
                      G4ThreeVector position = step->GetTrack()->GetPosition();
                      G4double globalTime = step->GetTrack()->GetGlobalTime();
                      G4int parentID = step->GetTrack()->GetTrackID();
                      G4double trackWeight = step->GetTrack()->GetWeight() / currentSecondaryMultiplicity;
	    
                      // Main loop dedicated to secondary hybrid particle 
                      for(int i=0; i<currentSecondaryMultiplicity; i++)
                        {
                          // Random generation of the angle (no physical crossSection for fluorescence)
                          double phi = G4RandFlat::shoot(twopi);
                          double cosTheta = G4RandFlat::shoot(-1.0, 1.0);
                          G4ThreeVector momentum;
                          momentum.setX(cos(phi)*sqrt(1.-(cosTheta*cosTheta)));
                          momentum.setY(sin(phi)*sqrt(1.-(cosTheta*cosTheta)));
                          momentum.setZ(cosTheta);

                          if(mIsHybridinoEnabled)
                            {
                              // Create a hybrid track and attach it to the primary particle
                              G4DynamicParticle *hybridParticle = new G4DynamicParticle(mHybridino, momentum, energy);
                              G4Track *newTrack = new G4Track(hybridParticle, globalTime, position);  
                              newTrack->SetParentID(parentID);
                              trackVector->push_back(newTrack);
		
                              // Store the hybrid particle weight and track for exponential attenuation step
                              mListOfHybridTrack.push_back(newTrack);
                              mListOfHybridWeight.push_back(trackWeight);
                            }
                          else
                            {
                              // create a hybrid struct for raycasting
                              // primary or not - energy - weight - position - direction
                              mListOfRaycasting.push_back(RaycastingStruct(false, energy, trackWeight, position, momentum));
                            }
                        }
                    }
                }
            }
          else
            {
              // Duplication of the step, track and particle (avoid the modification of the real step)
              // This step (Energy, Momentum and Position) reflects the status of the particle before the interaction 
              G4double incidentEnergy = step->GetPreStepPoint()->GetKineticEnergy();
              G4ThreeVector incidentMomentum = step->GetPreStepPoint()->GetMomentumDirection();
              G4ThreeVector incidentPosition = step->GetPostStepPoint()->GetPosition();
	
              G4DynamicParticle *myIncidentPhoton = new G4DynamicParticle(G4Gamma::Gamma(),incidentMomentum,incidentEnergy);
              G4Track *myTrack = new G4Track(myIncidentPhoton,0.,incidentPosition);
              myTrack->SetMomentumDirection(incidentMomentum);
              G4Step *myStep = new G4Step();
              myStep->SetTrack(myTrack);
              myStep->SetPreStepPoint(new G4StepPoint(*step->GetPreStepPoint()));
              myStep->SetPostStepPoint(new G4StepPoint(*step->GetPostStepPoint()));
              myStep->GetPostStepPoint()->SetMomentumDirection(incidentMomentum);
              myStep->SetStepLength(step->GetStepLength());

              //       GateMessage("Actor", 0, "prePos = " << myStep->GetPreStepPoint()->GetPosition() << " preDir = " << myStep->GetPreStepPoint()->GetMomentumDirection() << Gateendl);
              //       GateMessage("Actor", 0, "posPos = " << myStep->GetPostStepPoint()->GetPosition() << " posDir = " << myStep->GetPostStepPoint()->GetMomentumDirection() << Gateendl);
              //       GateMessage("Actor", 0, "traPos = " << myStep->GetTrack()->GetPosition() << " traDir = " << myStep->GetTrack()->GetMomentumDirection() << " trackAdress = " << myStep->GetTrack() << Gateendl);

              int currentSecondaryMultiplicity;
              std::map<G4VPhysicalVolume *,int>::iterator it = mSecondaryMultiplicityMap.find(step->GetTrack()->GetVolume());
              if(it == mSecondaryMultiplicityMap.end()) { currentSecondaryMultiplicity = mDefaultSecondaryMultiplicity; }
              else { currentSecondaryMultiplicity = it->second; }

              G4ThreeVector position = step->GetTrack()->GetPosition();
              G4double globalTime = step->GetTrack()->GetGlobalTime();
              G4int parentID = step->GetTrack()->GetTrackID();
              G4double trackWeight = step->GetTrack()->GetWeight() / currentSecondaryMultiplicity;
              G4VParticleChange* particleChange(0);
              G4TrackVector *trackVector = (const_cast<G4Step *>(step))->GetfSecondary();

              // Main loop dedicated to secondary hybrid particle 
              for(int i=0; i<currentSecondaryMultiplicity; i++)
                {
                  // Call the PostStepDoIt function related to the current process
                  particleChange = currentProcess->PostStepDoIt((const G4Track)(*myTrack), *myStep);
                  particleChange->SetVerboseLevel(0);
                  particleChange->UpdateStepForPostStep(myStep);

                  // 	GateMessage("Actor", 0, "prePos = " << newStep->GetPreStepPoint()->GetPosition() << " preDir = " << newStep->GetPreStepPoint()->GetMomentumDirection() << Gateendl);
                  // 	GateMessage("Actor", 0, "posPos = " << newStep->GetPostStepPoint()->GetPosition() << " posDir = " << newStep->GetPostStepPoint()->GetMomentumDirection() << Gateendl);

                  G4double energy = myStep->GetPostStepPoint()->GetKineticEnergy();
                  G4ThreeVector momentum = myStep->GetPostStepPoint()->GetMomentumDirection();

                  if(mIsHybridinoEnabled)
                    {
                      // Create a hybrid track and attach it to the primary particle
                      G4DynamicParticle *hybridParticle = new G4DynamicParticle(mHybridino, momentum, energy);
                      G4Track *newTrack = new G4Track(hybridParticle, globalTime, position);  
                      newTrack->SetParentID(parentID);
                      trackVector->push_back(newTrack);
	    
                      // Store the hybrid particle weight and track for exponential attenuation step
                      mListOfHybridTrack.push_back(newTrack);
                      mListOfHybridWeight.push_back(trackWeight);
                    }
                  else
                    {
                      // create a hybrid struct for raycasting
                      // primary or not - energy - weight - position - direction
                      mListOfRaycasting.push_back(RaycastingStruct(false, energy, trackWeight, position, momentum));
                    }
                }
	
              delete myTrack;
              delete myStep;
            }
        } 
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateSETLEMultiplicityActor::SaveData() {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEMultiplicityActor::ResetData() {}
//-----------------------------------------------------------------------------

GateSETLEMultiplicityActor *GateSETLEMultiplicityActor::singleton_SETLEMultiplicityActor = 0;

#endif /* end #define GATESIMULATIONSTATISTICACTOR_CC */

