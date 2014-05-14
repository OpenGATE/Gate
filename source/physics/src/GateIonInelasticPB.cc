/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateIonInelasticPB.hh"

#include "GateHadronicStandardProcessMessenger.hh"

//!!!!!!!!!!!!!!!!!!!!!!!!!!! He3? !!!!!!!!!!!!!!!!!!!!!!!!!!

//-----------------------------------------------------------------------------
GateIonInelasticPB::GateIonInelasticPB():GateVProcess("IonInelastic")
{  
  SetDefaultParticle("GenericIon");
  SetDefaultParticle("alpha");
  SetDefaultParticle("deuteron");
  SetDefaultParticle("triton");

  SetProcessInfo("Inelastic scattering for ions");

  AddToModelList("G4BinaryLightIonReaction");
  AddToModelList("G4WilsonAbrasionModel");
  AddToModelList("G4LEDeuteronInelastic");
  AddToModelList("G4LETritonInelastic");
  AddToModelList("G4LEAlphaInelastic");
  AddToModelList("G4QMDReaction");

  AddToDataSetList("G4TripathiCrossSection");
  AddToDataSetList("G4IonsKoxCrossSection");
  AddToDataSetList("G4IonsShenCrossSection");
  AddToDataSetList("G4IonsSihverCrossSection");
  AddToDataSetList("G4TripathiLightCrossSection");
  AddToDataSetList("G4HadronInelasticDataSet");

  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateIonInelasticPB::CreateProcess(G4ParticleDefinition * par)
{
  if(par==G4GenericIon::GenericIon() ) return new G4HadronInelasticProcess(GetG4ProcessName(),par);
  if(par==G4Triton::Triton() ) return new G4TritonInelasticProcess(GetG4ProcessName());
  if(par==G4Alpha::Alpha() ) return new G4AlphaInelasticProcess(GetG4ProcessName());
  if(par==G4Deuteron::Deuteron() ) return new G4DeuteronInelasticProcess(GetG4ProcessName());
  else {
    GateError("Error in GateIonInelasticPB::CreateProcess particle should be GenericIon/Triton/Alpha/Deuteron " 
	      << "\n while it is " << par->GetParticleName() 
	      << G4endl);
    return NULL;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIonInelasticPB::ConstructProcess( G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());        
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonInelasticPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4GenericIon::GenericIon()
     || par==G4Triton::Triton()
     || par==G4Alpha::Alpha()
     || par==G4Deuteron::Deuteron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonInelasticPB::IsModelApplicable(G4String model,G4ParticleDefinition * par)
{
  if( (model=="G4BinaryLightIonReaction" || model=="G4QMDReaction") && 
      (par == G4GenericIon::GenericIon()
       || par==G4Triton::Triton()
       || par==G4Alpha::Alpha()
       || par==G4Deuteron::Deuteron())) return true;
  else if(par == G4GenericIon::GenericIon() && model == "G4WilsonAbrasionModel") return true;
  else if(par == G4Triton::Triton() && model == "G4LETritonInelastic" ) return true;
  else if(par == G4Deuteron::Deuteron() && model == "G4LEDeuteronInelastic" ) return true; 
  else if(par == G4Alpha::Alpha() && model == "G4LEAlphaInelastic" ) return true; 
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonInelasticPB::IsDatasetApplicable(G4String cs,G4ParticleDefinition * par)
{
  if(par == G4GenericIon::GenericIon() &&
      (cs == "G4TripathiCrossSection"
       || cs == "G4IonsKoxCrossSection"
       || cs == "G4IonsShenCrossSection"
       || cs == "G4IonsSihverCrossSection" )) return true;
  else if( (cs == "G4TripathiLightCrossSection" 
            || cs == "G4HadronInelasticDataSet") &&
	  (par == G4Triton::Triton()
           || par == G4Deuteron::Deuteron()
           || par == G4Alpha::Alpha() )) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIonInelasticPB::AddUserDataSet(G4String ){}
void GateIonInelasticPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------



MAKE_PROCESS_AUTO_CREATOR_CC(GateIonInelasticPB)
