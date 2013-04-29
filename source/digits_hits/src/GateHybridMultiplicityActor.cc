/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateHybridMultiplicityActor : 
  \brief 
*/

#ifndef GATEHYBRIDMULTIPLICITYACTOR_CC
#define GATEHYBRIDMULTIPLICITYACTOR_CC

#include "GateHybridMultiplicityActor.hh"
#include "GateHybridDoseActor.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"
#include "GateActorManager.hh"
#include "GateDetectorConstruction.hh"
#include "GateMultiSensitiveDetector.hh"
#include "G4Event.hh"
#include "G4Hybridino.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateHybridMultiplicityActor::GateHybridMultiplicityActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateHybridMultiplicityActor() -- begin"<<G4endl);
  materialHandler = new GateMaterialMuHandler(100);
//   pActor = new GateHybridMultiplicityActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateHybridMultiplicityActor() -- end"<<G4endl);

  defaultPrimaryMultiplicity = 2;
  defaultSecondaryMultiplicity = 4;
//   defaultPrimaryMultiplicity = -1;
//   defaultSecondaryMultiplicity = -1;
  secondaryMultiplicityMap.clear();
  processListForGamma = 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor 
GateHybridMultiplicityActor::~GateHybridMultiplicityActor() 
{
//   delete pActor;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateHybridMultiplicityActor::Construct()
{
  GateMessage("Actor", 0, " HybridMultiplicityActor auto-construction" << G4endl);
  
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
    
  if((defaultPrimaryMultiplicity<0) or (defaultSecondaryMultiplicity<0)) {
    GateError("Please set primary and secondary multiplicities for 'HybridMultiplicityActor'");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor 
int GateHybridMultiplicityActor::AddSecondaryMultiplicity(G4VPhysicalVolume *v) 
{
  G4MultiFunctionalDetector *G4MSD = dynamic_cast<GateMultiSensitiveDetector *>(v->GetLogicalVolume()->GetSensitiveDetector())->GetMultiFunctionalDetector();
  
  int numberOfActor = G4MSD->GetNumberOfPrimitives();
  int numberOfHybridDoseActor = 0;
  GateHybridDoseActor *hybridDoseActor = 0;
  for(int i=0; i<numberOfActor; i++)
  {
    GateVActor *gateActor = dynamic_cast<GateVActor *>(G4MSD->GetPrimitive(i));
    if(gateActor->GetTypeName() == "GateHybridDoseActor")
    {
      hybridDoseActor = dynamic_cast<GateHybridDoseActor *>(gateActor);
      numberOfHybridDoseActor++;
    }
  }
  
  if(numberOfHybridDoseActor == 0) {
    secondaryMultiplicityMap.insert(make_pair(v->GetName(),defaultSecondaryMultiplicity));
    return defaultSecondaryMultiplicity;
  }
  else if(numberOfHybridDoseActor == 1) {
    secondaryMultiplicityMap.insert(make_pair(v->GetName(),hybridDoseActor->GetSecondaryMultiplicity()));
    return hybridDoseActor->GetSecondaryMultiplicity();
  }
  else {
    GateError("The number of 'hybridDoseActor' attached to '" << hybridDoseActor->GetVolumeName() << "' is too large (1 maximum)");
    return 0;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Event
void GateHybridMultiplicityActor::BeginOfEventAction(const G4Event *event)
{
  if(!processListForGamma) { processListForGamma = G4Gamma::Gamma()->GetProcessManager()->GetProcessList(); }
  
  GateVSource* source = GateSourceMgr::GetInstance()->GetSource(GateSourceMgr::GetInstance()->GetCurrentSourceID());
  if(source->GetParticleDefinition()->GetParticleName() == "gamma")
  {
    G4ParticleDefinition *hybridino = G4Hybridino::Hybridino();
    G4ParticleDefinition *gamma = G4Gamma::Gamma();

    GateSPSEneDistribution* enedist = source->GetEneDist();
    GateSPSPosDistribution* posdist = source->GetPosDist();
    GateSPSAngDistribution* angdist = source->GetAngDist();

    G4Event *modifiedEvent = const_cast<G4Event *>(event);
    // WARNING - G4Event cannot be modified by default because of its 'const' status.
    // Use of the 'const_cast' function to overcome this problem.

    G4ThreeVector position;
    G4ThreeVector momentum;
    G4double energy;
    for(int i=0; i<defaultPrimaryMultiplicity; i++)
    {
      position = posdist->GenerateOne();
      momentum = angdist->GenerateOne();
      energy = enedist->GenerateOne(gamma);

      G4PrimaryParticle *hybridParticle = new G4PrimaryParticle(hybridino,momentum.x(),momentum.y(),momentum.z(),energy);
      hybridParticle->SetKineticEnergy(energy*MeV);

      G4PrimaryVertex *hybridVertex = new G4PrimaryVertex();
      hybridVertex->SetPosition(position.x(),position.y(),position.z());
      hybridVertex->SetPrimary(hybridParticle);

      modifiedEvent->AddPrimaryVertex(hybridVertex); 
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridMultiplicityActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  if(t->GetParticleDefinition()->GetParticleName() == "hybridino")
  {
//     GateMessage("Actor", 0, "track = " << t << " parentID = " << t->GetParentID() << G4endl);
    
    currentHybridTrackWeight = 1.;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateHybridMultiplicityActor::UserSteppingAction(const GateVVolume *, const G4Step * step)
{
  DD("Multiplicity step");
  
  G4String particleName = step->GetTrack()->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();
    if(particleName == "hybridino")
  {
    G4double stepLength = step->GetStepLength();
    if(stepLength > 0.0)
    {
      G4Material *material = step->GetPreStepPoint()->GetMaterial();
      G4double energy = step->GetPreStepPoint()->GetKineticEnergy();
      G4double mu = materialHandler->GetMu(material, energy)*material->GetDensity()/(g/cm3);    
      currentHybridTrackWeight = currentHybridTrackWeight * exp(-mu*stepLength/10.);
    }
  }
  else if(particleName == "gamma")
  {
    G4String processName = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();    
    if(processName != "Transportation")
    {
      G4VProcess *currentProcess = 0;
      for(int i=0; i<processListForGamma->size(); i++)
      {
	if((*processListForGamma)[i]->GetProcessName() == processName)
	{
	  currentProcess = (*processListForGamma)[i];
	  break;
	}      
      }
      if(currentProcess == 0) { GateError("the process doesn't exist"); }
      
      if(processName == "PhotoElectric") {}
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

  //       GateMessage("Actor", 0, "prePos = " << myStep->GetPreStepPoint()->GetPosition() << " preDir = " << myStep->GetPreStepPoint()->GetMomentumDirection() << G4endl);
  //       GateMessage("Actor", 0, "posPos = " << myStep->GetPostStepPoint()->GetPosition() << " posDir = " << myStep->GetPostStepPoint()->GetMomentumDirection() << G4endl);
  //       GateMessage("Actor", 0, "traPos = " << myStep->GetTrack()->GetPosition() << " traDir = " << myStep->GetTrack()->GetMomentumDirection() << " trackAdress = " << myStep->GetTrack() << G4endl);

	G4VPhysicalVolume *currentPhysicalVolume = step->GetTrack()->GetVolume();
	int currentSecondaryMultiplicity;
	std::map<G4String,int>::iterator it = secondaryMultiplicityMap.find(currentPhysicalVolume->GetName());
	if(it == secondaryMultiplicityMap.end()) { currentSecondaryMultiplicity = AddSecondaryMultiplicity(currentPhysicalVolume); }
	else { currentSecondaryMultiplicity = it->second; }

	G4ParticleDefinition *hybridino = G4Hybridino::Hybridino();
	G4VParticleChange* particleChange(0);
	std::vector<G4DynamicParticle *> secondaries;
	secondaries.reserve(currentSecondaryMultiplicity);
	G4Step *newStep;

	// Call Ms times the PostStepDoIt function and store a list of secondary hybridinos 
	for(int i=0; i<currentSecondaryMultiplicity; i++)
	{
	  particleChange = currentProcess->PostStepDoIt((const G4Track)(*myTrack), *myStep);
	  particleChange->SetVerboseLevel(0);
	  newStep = particleChange->UpdateStepForPostStep(myStep);

  // 	GateMessage("Actor", 0, "prePos = " << newStep->GetPreStepPoint()->GetPosition() << " preDir = " << newStep->GetPreStepPoint()->GetMomentumDirection() << G4endl);
  // 	GateMessage("Actor", 0, "posPos = " << newStep->GetPostStepPoint()->GetPosition() << " posDir = " << newStep->GetPostStepPoint()->GetMomentumDirection() << G4endl);
	  
	  G4double energy = newStep->GetPostStepPoint()->GetKineticEnergy();
	  G4ThreeVector momentum = newStep->GetPostStepPoint()->GetMomentumDirection();
	  G4DynamicParticle *hybridParticle = new G4DynamicParticle(hybridino, momentum, energy);
	  secondaries.push_back(hybridParticle);
	}

	// Attach the list of secondary hybridinos to the primary particle
	G4TrackVector *trackVector = (const_cast<G4Step *>(step))->GetfSecondary();
	// WARNING - G4Event cannot be modified by default because of its 'const' status.
	// Use of the 'const_cast' function to overcome this problem.
	std::vector<G4DynamicParticle*>::iterator iter = secondaries.begin();
	while (iter != secondaries.end())
	{
	  G4DynamicParticle* particle = *iter;
	  G4Track *newTrack = new G4Track(particle, step->GetTrack()->GetGlobalTime(), step->GetTrack()->GetPosition());
	  newTrack->SetParentID(step->GetTrack()->GetTrackID());
	  trackVector->push_back(newTrack); 
	  iter++;
	  
  // 	GateMessage("Actor", 0, "traPos = " << newTrack->GetPosition() << " traDir = " << newTrack->GetMomentumDirection() << " trackAdress = " << newTrack << G4endl);
	}
	
	delete myTrack;
	delete myStep;
      }
  //     G4TrackVector *trackVector = (const_cast<G4Step *>(step))->GetfSecondary();
  //     GateMessage("Actor", 0, "trackNumber = " << trackVector->size() << G4endl);
    } 
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateHybridMultiplicityActor::SaveData() {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridMultiplicityActor::ResetData() {}
//-----------------------------------------------------------------------------

GateHybridMultiplicityActor *GateHybridMultiplicityActor::singleton_HybridMultiplicityActor = 0;

#endif /* end #define GATESIMULATIONSTATISTICACTOR_CC */

