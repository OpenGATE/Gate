/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
  
  
#ifndef GateMixedDNAPhysics_h
#define GateMixedDNAPhysics_h 1

#include "G4VUserPhysicsList.hh"
#include "G4VModularPhysicsList.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTypes.hh"
#include "G4VPhysicsConstructor.hh"
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateMixedDNAPhysicsMessenger;

class GateMixedDNAPhysics: public G4VUserPhysicsList
{
public:

  GateMixedDNAPhysics(G4String);
  virtual ~GateMixedDNAPhysics();
public:
//  void defineRegionsWithDNA(G4String);
  void defineRegionsWithDNA(G4String nameRegions) {
	regionsWithDNA.push_back(nameRegions);

  }
  
protected:

  // these methods construct particles 
  void ConstructBosons();
  void ConstructLeptons();
  void ConstructBarions();

  // these methods construct physics processes and register them
  void ConstructEM();

  // Construct particle and physics
  void ConstructParticle();
  void ConstructProcess();


  void setDefaultModelsInWorld(G4String);
  void setDNAInWorld();
  void inactivateDefaultModelsInRegion();
  void activateDNAInRegion();
    
private:

  GateMixedDNAPhysicsMessenger *pMessenger;   //
  G4String nameprocessmixed;		//
  std::vector <G4String> regionsWithDNA;   // vector for regions with DNA
 


};
#endif





