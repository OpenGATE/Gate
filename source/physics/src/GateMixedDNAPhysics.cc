/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "G4Version.hh"
#include "G4SystemOfUnits.hh"

#include "GateMixedDNAPhysics.hh"
#include "GateMixedDNAPhysicsMessenger.hh"
#include <vector>
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateMixedDNAPhysics::GateMixedDNAPhysics(G4String nameProcessMixed):  G4VUserPhysicsList()

{
  SetVerboseLevel(1);

  regionsWithDNA.clear();
  nameprocessmixed = nameProcessMixed;

  pMessenger = new GateMixedDNAPhysicsMessenger(this);
  pMessenger->BuildCommands("/gate/physics");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateMixedDNAPhysics::~GateMixedDNAPhysics()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateMixedDNAPhysics::ConstructParticle()
{
  ConstructBosons();
  ConstructLeptons();
  ConstructBarions();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateMixedDNAPhysics::ConstructBosons()
{
  // gamma
  G4Gamma::GammaDefinition();
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateMixedDNAPhysics::ConstructLeptons()
{
  // leptons
  G4Electron::ElectronDefinition();
  G4Positron::PositronDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

//DNA
#include "G4DNAGenericIonsManager.hh"
//ENDDNA

void GateMixedDNAPhysics::ConstructBarions()
{
  //  baryons
  G4Proton::ProtonDefinition();
  G4GenericIon::GenericIonDefinition();

  // Geant4 DNA new particles
  G4DNAGenericIonsManager * genericIonsManager;
  genericIonsManager=G4DNAGenericIonsManager::Instance();
  genericIonsManager->GetIon("alpha++");
  genericIonsManager->GetIon("alpha+");
  genericIonsManager->GetIon("helium");
  genericIonsManager->GetIon("hydrogen");
  genericIonsManager->GetIon("carbon");
  genericIonsManager->GetIon("nitrogen");
  genericIonsManager->GetIon("oxygen");
  genericIonsManager->GetIon("iron");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateMixedDNAPhysics::ConstructProcess()
{
  ConstructEM();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

// Geant4-DNA MODELS

#include "G4DNAElastic.hh"
#include "G4DNAChampionElasticModel.hh"
#include "G4DNAScreenedRutherfordElasticModel.hh"

#include "G4DNAExcitation.hh"
#include "G4DNAMillerGreenExcitationModel.hh"
#include "G4DNABornExcitationModel.hh"

#include "G4DNAIonisation.hh"
#include "G4DNABornIonisationModel.hh"
#include "G4DNARuddIonisationModel.hh"

#include "G4DNAChargeDecrease.hh"
#include "G4DNADingfelderChargeDecreaseModel.hh"

#include "G4DNAChargeIncrease.hh"
#include "G4DNADingfelderChargeIncreaseModel.hh"

#include "G4DNAAttachment.hh"
#include "G4DNAMeltonAttachmentModel.hh"

#include "G4DNAVibExcitation.hh"
#include "G4DNASancheExcitationModel.hh"

//

#include "G4LossTableManager.hh"
#include "G4EmConfigurator.hh"
#include "G4VEmModel.hh"
#include "G4DummyModel.hh"
#include "G4eIonisation.hh"
#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"
#include "G4eMultipleScattering.hh"
#include "G4hMultipleScattering.hh"
#include "G4BraggIonGasModel.hh"
#include "G4BetheBlochIonGasModel.hh"
#if G4VERSION_NUMBER >= 1000
#include "G4UrbanMscModel.hh"
#else
#include "G4UrbanMscModel93.hh"
#endif
#include "G4MollerBhabhaModel.hh"
#include "G4IonFluctuations.hh"
#include "G4UniversalFluctuation.hh"


// gamma
#include "G4Gamma.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"
#include "G4ComptonScattering.hh"
#include "G4LivermoreComptonModel.hh"
#include "G4GammaConversion.hh"
#include "G4LivermoreGammaConversionModel.hh"
#include "G4RayleighScattering.hh"
#include "G4LivermoreRayleighModel.hh"



// Livermore electrons

#include "G4LivermoreBremsstrahlungModel.hh"
#include "G4LivermoreIonisationModel.hh"
#include "G4eBremsstrahlung.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....




////////////////////////////////////////////////////////////////////////////////////////////////

// ConstructEM
void GateMixedDNAPhysics::ConstructEM()
{
  setDefaultModelsInWorld(nameprocessmixed);
  setDNAInWorld();
  inactivateDefaultModelsInRegion();
  activateDNAInRegion();
}

// SetDefaultModel
void GateMixedDNAPhysics::setDefaultModelsInWorld(G4String NPM) {
  if (NPM == "emstandard_opt3_mixed_emdna") {

    theParticleIterator->reset();
    while( (*theParticleIterator)() )
      {
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        // *********************************
        // 1) Processes for the World region
        // *********************************

        if (particleName == "e-") {

          // STANDARD msc is active in the world
          G4eMultipleScattering* msc = new G4eMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4eIonisation* eion = new G4eIonisation();
          eion->SetEmModel(new G4MollerBhabhaModel(), 1);
          pmanager->AddProcess(eion, -1, 2, 2);

        } else if ( particleName == "proton" ) {

          // STANDARD msc is active in the world
          G4hMultipleScattering* msc = new G4hMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4hIonisation* hion = new G4hIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(), 1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if (particleName == "GenericIon") { // THIS IS NEEDED FOR STANDARD ALPHA G4ionIonisation PROCESS

          // STANDARD msc is active in the world
          pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4ionIonisation* hion = new G4ionIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(),1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if ( particleName == "alpha" ) {

          // STANDARD msc is active in the world
          G4hMultipleScattering* msc = new G4hMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4ionIonisation* hion = new G4ionIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(),1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if (particleName == "gamma") {

          G4double LivermoreHighEnergyLimit = GeV;

          G4PhotoElectricEffect* thePhotoElectricEffect = new G4PhotoElectricEffect();
          G4LivermorePhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePhotoElectricModel();
          theLivermorePhotoElectricModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          thePhotoElectricEffect->AddEmModel(0, theLivermorePhotoElectricModel);
          pmanager->AddDiscreteProcess(thePhotoElectricEffect);

          G4ComptonScattering* theComptonScattering = new G4ComptonScattering();
          G4LivermoreComptonModel* theLivermoreComptonModel = new G4LivermoreComptonModel();
          theLivermoreComptonModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theComptonScattering->AddEmModel(0, theLivermoreComptonModel);
          pmanager->AddDiscreteProcess(theComptonScattering);

          G4GammaConversion* theGammaConversion = new G4GammaConversion();
          G4LivermoreGammaConversionModel* theLivermoreGammaConversionModel = new G4LivermoreGammaConversionModel();
          theLivermoreGammaConversionModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theGammaConversion->AddEmModel(0, theLivermoreGammaConversionModel);
          pmanager->AddDiscreteProcess(theGammaConversion);

          G4RayleighScattering* theRayleigh = new G4RayleighScattering();
          G4LivermoreRayleighModel* theRayleighModel = new G4LivermoreRayleighModel();
          theRayleighModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theRayleigh->AddEmModel(0, theRayleighModel);
          pmanager->AddDiscreteProcess(theRayleigh);
        }
      }
  } else if (NPM == "emlivermore_mixed_emdna") {

    theParticleIterator->reset();
    while( (*theParticleIterator)() )
      {
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        // *********************************
        // 1) Processes LIVERMORE for the World region
        // *********************************

        if (particleName == "e-") {

          // LIVERMORE msc is active in the world
          G4eMultipleScattering* msc = new G4eMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // LIVERMORE ionisation is active in the world
          G4eIonisation* eion = new G4eIonisation();
          eion->SetEmModel(new G4LivermoreIonisationModel(),1);         //   G4LivermoreIonisationModel
          // eion->SetEmModel(new G4MollerBhabhaModel(), 1);
          pmanager->AddProcess(eion, -1, 2, 2);

          // LIVERMORE Bremsstrahlung is active in the world
          G4eBremsstrahlung* eBrem = new G4eBremsstrahlung();
          eBrem->SetEmModel(new G4LivermoreBremsstrahlungModel(), 1);   //  G4LivermoreBremsstrahlungModel G4LivermoreIonisationModel
          pmanager->AddProcess(eion, -1, 2, 2);


        } else if ( particleName == "proton" ) {			//identique avec standard opt3

          // STANDARD msc is active in the world
          G4hMultipleScattering* msc = new G4hMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4hIonisation* hion = new G4hIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(), 1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if (particleName == "GenericIon") { 				//identique avec standard opt3

          // THIS IS NEEDED FOR STANDARD ALPHA G4ionIonisation PROCESS
          // STANDARD msc is active in the world
          pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4ionIonisation* hion = new G4ionIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(),1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if ( particleName == "alpha" ) {	//indentique avec standard opt3

          // STANDARD msc is active in the world
          G4hMultipleScattering* msc = new G4hMultipleScattering();
          pmanager->AddProcess(msc, -1, 1, 1);

          // STANDARD ionisation is active in the world
          G4ionIonisation* hion = new G4ionIonisation();
          hion->SetEmModel(new G4BraggIonGasModel(),1);
          hion->SetEmModel(new G4BetheBlochIonGasModel(), 2);
          pmanager->AddProcess(hion, -1, 2, 2);

        } else if (particleName == "gamma") {         //gamme Livermore indentique avec standard opt3

          G4double LivermoreHighEnergyLimit = GeV;

          G4PhotoElectricEffect* thePhotoElectricEffect = new G4PhotoElectricEffect();
          G4LivermorePhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePhotoElectricModel();
          theLivermorePhotoElectricModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          thePhotoElectricEffect->AddEmModel(0, theLivermorePhotoElectricModel);
          pmanager->AddDiscreteProcess(thePhotoElectricEffect);

          G4ComptonScattering* theComptonScattering = new G4ComptonScattering();
          G4LivermoreComptonModel* theLivermoreComptonModel = new G4LivermoreComptonModel();
          theLivermoreComptonModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theComptonScattering->AddEmModel(0, theLivermoreComptonModel);
          pmanager->AddDiscreteProcess(theComptonScattering);

          G4GammaConversion* theGammaConversion = new G4GammaConversion();
          G4LivermoreGammaConversionModel* theLivermoreGammaConversionModel = new G4LivermoreGammaConversionModel();
          theLivermoreGammaConversionModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theGammaConversion->AddEmModel(0, theLivermoreGammaConversionModel);
          pmanager->AddDiscreteProcess(theGammaConversion);

          G4RayleighScattering* theRayleigh = new G4RayleighScattering();
          G4LivermoreRayleighModel* theRayleighModel = new G4LivermoreRayleighModel();
          theRayleighModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
          theRayleigh->AddEmModel(0, theRayleighModel);
          pmanager->AddDiscreteProcess(theRayleigh);
        }
      }

  } else G4cout << "No processes" << G4endl;
}

// Set DNA In World
void GateMixedDNAPhysics::setDNAInWorld() {

  theParticleIterator->reset();
  while( (*theParticleIterator)() )
    {
      G4ParticleDefinition* particle = theParticleIterator->value();
      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4String particleName = particle->GetParticleName();
      // *********************************
      // 1) Processes for the World region
      // *********************************

      if (particleName == "e-") {

        // DNA elastic is not active in the world
        G4DNAElastic* theDNAElasticProcess = new G4DNAElastic("e-_G4DNAElastic");
#if G4VERSION_NUMBER >= 960
        theDNAElasticProcess->SetEmModel(new G4DummyModel(),1);
#else
        theDNAElasticProcess->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(theDNAElasticProcess);

        // DNA excitation is not active in the world
        G4DNAExcitation* dnaex = new G4DNAExcitation("e-_G4DNAExcitation");
#if G4VERSION_NUMBER >= 960
        dnaex->SetEmModel(new G4DummyModel(),1);
#else
        dnaex->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnaex);

        // DNA ionisation is not active in the world
        G4DNAIonisation* dnaioni = new G4DNAIonisation("e-_G4DNAIonisation");
#if G4VERSION_NUMBER >= 960
        dnaioni->SetEmModel(new G4DummyModel(),1);
#else
        dnaioni->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnaioni);

        // DNA attachment is not active in the world
        G4DNAAttachment* dnaatt = new G4DNAAttachment("e-_G4DNAAttachment");
#if G4VERSION_NUMBER >= 960
        dnaatt->SetEmModel(new G4DummyModel(),1);
#else
        dnaatt->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnaatt);

        // DNA vib. excitation is not active in the world
        G4DNAVibExcitation* dnavib = new G4DNAVibExcitation("e-_G4DNAVibExcitation");
#if G4VERSION_NUMBER >= 960
        dnavib->SetEmModel(new G4DummyModel(),1);
#else
        dnavib->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnavib);

      } else if ( particleName == "proton" ) {

        // DNA excitation is not active in the world
        G4DNAExcitation* dnaex = new G4DNAExcitation("proton_G4DNAExcitation");
#if G4VERSION_NUMBER >= 960
        dnaex->SetEmModel(new G4DummyModel(),1);
        dnaex->SetEmModel(new G4DummyModel(),2);
#else
        dnaex->SetModel(new G4DummyModel(),1);
        dnaex->SetModel(new G4DummyModel(),2);
#endif
        pmanager->AddDiscreteProcess(dnaex);

        // DNA ionisation is not active in the world
        G4DNAIonisation* dnaioni = new G4DNAIonisation("proton_G4DNAIonisation");
#if G4VERSION_NUMBER >= 960
        dnaioni->SetEmModel(new G4DummyModel(),1);
        dnaioni->SetEmModel(new G4DummyModel(),2);
#else
        dnaioni->SetModel(new G4DummyModel(),1);
        dnaioni->SetModel(new G4DummyModel(),2);
#endif
        pmanager->AddDiscreteProcess(dnaioni);

        // DNA charge decrease is ACTIVE in the world since no corresponding STANDARD process exist
        pmanager->AddDiscreteProcess(new G4DNAChargeDecrease("proton_G4DNAChargeDecrease"));

      } else if ( particleName == "hydrogen" ) {

        // DNA processes are ACTIVE in the world since no corresponding STANDARD processes exist
        pmanager->AddDiscreteProcess(new G4DNAIonisation("hydrogen_G4DNAIonisation"));
        pmanager->AddDiscreteProcess(new G4DNAExcitation("hydrogen_G4DNAExcitation"));
        pmanager->AddDiscreteProcess(new G4DNAChargeIncrease("hydrogen_G4DNAChargeIncrease"));

      } else if ( particleName == "alpha" ) {

        // DNA excitation is not active in the world
        G4DNAExcitation* dnaex = new G4DNAExcitation("alpha_G4DNAExcitation");
#if G4VERSION_NUMBER >= 960
        dnaex->SetEmModel(new G4DummyModel(),1);
#else
        dnaex->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnaex);

        // DNA ionisation is not active in the world
        G4DNAIonisation* dnaioni = new G4DNAIonisation("alpha_G4DNAIonisation");
#if G4VERSION_NUMBER >= 960
        dnaioni->SetEmModel(new G4DummyModel(),1);
#else
        dnaioni->SetModel(new G4DummyModel(),1);
#endif
        pmanager->AddDiscreteProcess(dnaioni);

        // DNA charge decrease is ACTIVE in the world since no corresponding STANDARD process exist
        pmanager->AddDiscreteProcess(new G4DNAChargeDecrease("alpha_G4DNAChargeDecrease"));

      } else if ( particleName == "alpha+" ) {

        // DNA processes are ACTIVE in the world since no corresponding STANDARD processes exist
        pmanager->AddDiscreteProcess(new G4DNAExcitation("alpha+_G4DNAExcitation"));
        pmanager->AddDiscreteProcess(new G4DNAIonisation("alpha+_G4DNAIonisation"));
        pmanager->AddDiscreteProcess(new G4DNAChargeDecrease("alpha+_G4DNAChargeDecrease"));
        pmanager->AddDiscreteProcess(new G4DNAChargeIncrease("alpha+_G4DNAChargeIncrease"));

      } else if ( particleName == "helium" ) {

        // DNA processes are ACTIVE in the world since no corresponding STANDARD processes exist
        pmanager->AddDiscreteProcess(new G4DNAExcitation("helium_G4DNAExcitation"));
        pmanager->AddDiscreteProcess(new G4DNAIonisation("helium_G4DNAIonisation"));
        pmanager->AddDiscreteProcess(new G4DNAChargeIncrease("helium_G4DNAChargeIncrease"));

      }
    }
}


// Inactivate Default Models In Region
void GateMixedDNAPhysics::inactivateDefaultModelsInRegion() {

  for (unsigned int k = 0; k < regionsWithDNA.size(); k++) {

    // **************************************
    // 2) Define processes for Target region
    // **************************************

    // STANDARD EM processes should be inactivated when corresponding DNA processes are used
    // - STANDARD EM e- processes are inactivated below 1 MeV
    // - STANDARD EM proton & alpha processes are inactivated below standEnergyLimit
    G4double standEnergyLimit = 9.9*MeV;
    //

    G4double massFactor = 1.0079/4.0026;
    G4EmConfigurator* em_config = G4LossTableManager::Instance()->EmConfigurator();
    G4VEmModel* mod;

    // *** e-
    // ---> STANDARD EM processes are inactivated below 1 MeV
#if G4VERSION_NUMBER >= 1000
    mod =  new G4UrbanMscModel();
#else
    mod =  new G4UrbanMscModel93();
#endif
    mod->SetActivationLowEnergyLimit(1*MeV);
    //em_config->SetExtraEmModel("e-","msc",mod,"Target");
    em_config->SetExtraEmModel("e-","msc",mod,regionsWithDNA[k]);


    mod = new G4MollerBhabhaModel();
    mod->SetActivationLowEnergyLimit(0.99*MeV);
    //em_config->SetExtraEmModel("e-","eIoni",mod,"Target",0.0,100*TeV, new G4UniversalFluctuation());
    em_config->SetExtraEmModel("e-","eIoni",mod,regionsWithDNA[k],0.0,100*TeV, new G4UniversalFluctuation());


    // *** proton
    // ---> STANDARD EM processes inactivated below standEnergyLimit
    // STANDARD msc is still active
    // Inactivate following STANDARD processes
    mod = new G4BraggIonGasModel();
    mod->SetActivationLowEnergyLimit(standEnergyLimit);
    //em_config->SetExtraEmModel("proton","hIoni",mod,"Target",0.0,2*MeV, new G4IonFluctuations());
    em_config->SetExtraEmModel("proton","hIoni",mod,regionsWithDNA[k],0.0,2*MeV, new G4IonFluctuations());


    mod = new G4BetheBlochIonGasModel();
    mod->SetActivationLowEnergyLimit(standEnergyLimit);
    //em_config->SetExtraEmModel("proton","hIoni",mod,"Target",2*MeV,100*TeV, new G4UniversalFluctuation());
    em_config->SetExtraEmModel("proton","hIoni",mod,regionsWithDNA[k],2*MeV,100*TeV, new G4UniversalFluctuation());


    // *** alpha
    // ---> STANDARD EM processes inactivated below standEnergyLimit
    // STANDARD msc is still active
    // Inactivate following STANDARD processes

    mod = new G4BraggIonGasModel();
    mod->SetActivationLowEnergyLimit(standEnergyLimit);
    //em_config->SetExtraEmModel("alpha","ionIoni",mod,"Target",0.0,2*MeV/massFactor, new G4IonFluctuations());
    em_config->SetExtraEmModel("alpha","ionIoni",mod,regionsWithDNA[k],0.0,2*MeV/massFactor, new G4IonFluctuations());


    mod = new G4BetheBlochIonGasModel();
    mod->SetActivationLowEnergyLimit(standEnergyLimit);
    //em_config->SetExtraEmModel("alpha","ionIoni",mod,"Target",2*MeV/massFactor,100*TeV, new G4UniversalFluctuation());
    em_config->SetExtraEmModel("alpha","ionIoni",mod,regionsWithDNA[k],2*MeV/massFactor,100*TeV, new G4UniversalFluctuation());
  }
}


// Activate DNA In Region
void GateMixedDNAPhysics::activateDNAInRegion() {
  for (unsigned int k = 0; k < regionsWithDNA.size(); k++) {

    // **************************************
    // 2) Define processes for Target region
    // **************************************
    // STANDARD EM processes should be inactivated when corresponding DNA processes are used
    G4EmConfigurator* em_config = G4LossTableManager::Instance()->EmConfigurator();
    G4VEmModel* mod;

    // *** e-
    // ---> STANDARD EM processes are inactivated below 1 MeV
    // ---> DNA processes activated
    mod = new G4DNAChampionElasticModel();
    em_config->SetExtraEmModel("e-","e-_G4DNAElastic",mod,regionsWithDNA[k],0.0,1*MeV);


    mod = new G4DNABornIonisationModel();
    em_config->SetExtraEmModel("e-","e-_G4DNAIonisation",mod,regionsWithDNA[k],11*eV,1*MeV);


    mod = new G4DNABornExcitationModel();
    em_config->SetExtraEmModel("e-","e-_G4DNAExcitation",mod,regionsWithDNA[k],9*eV,1*MeV);


    mod = new G4DNAMeltonAttachmentModel();
    em_config->SetExtraEmModel("e-","e-_G4DNAAttachment",mod,regionsWithDNA[k],4*eV,13*eV);


    mod = new G4DNASancheExcitationModel();
    em_config->SetExtraEmModel("e-","e-_G4DNAVibExcitation",mod,regionsWithDNA[k],2*eV,100*eV);



    // *** proton
    // ---> DNA processes activated
    mod = new G4DNARuddIonisationModel();
    em_config->SetExtraEmModel("proton","proton_G4DNAIonisation",mod,regionsWithDNA[k],0.0,0.5*MeV);


    mod = new G4DNABornIonisationModel();
    em_config->SetExtraEmModel("proton","proton_G4DNAIonisation",mod,regionsWithDNA[k],0.5*MeV,10*MeV);


    mod = new G4DNAMillerGreenExcitationModel();
    em_config->SetExtraEmModel("proton","proton_G4DNAExcitation",mod,regionsWithDNA[k],10*eV,0.5*MeV);


    mod = new G4DNABornExcitationModel();
    em_config->SetExtraEmModel("proton","proton_G4DNAExcitation",mod,regionsWithDNA[k],0.5*MeV,10*MeV);


    // *** alpha
    // ---> DNA processes activated
    mod = new G4DNARuddIonisationModel();
    em_config->SetExtraEmModel("alpha","alpha_G4DNAIonisation",mod,regionsWithDNA[k],0.0,10*MeV);


    mod = new G4DNAMillerGreenExcitationModel();
    em_config->SetExtraEmModel("alpha","alpha_G4DNAExcitation",mod,regionsWithDNA[k],1*keV,10*MeV);
  }
}
