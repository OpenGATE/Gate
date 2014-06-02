/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateEmCalculatorActor :
  \brief
*/

#include "GateEmCalculatorActor.hh"
#include "GateMiscFunctions.hh"
#include "G4Event.hh"
#include "G4MaterialTable.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4UnitsTable.hh"
#include "G4ProductionCutsTable.hh"



//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateEmCalculatorActor::GateEmCalculatorActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateEmCalculatorActor() -- begin"<<G4endl);
  //SetTypeName("EmCalculatorActor");
//  pActor = new GateActorMessenger(this);
  ResetData();
  GateDebugMessageDec("Actor",4,"GateEmCalculatorActor() -- end"<<G4endl);

  mEnergy = 100 ;
  mPartName = "proton";

  pActorMessenger = new GateEmCalculatorActorMessenger(this);
  emcalc = new G4EmCalculator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateEmCalculatorActor::~GateEmCalculatorActor()
{
//  delete pActor;
  delete pActorMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateEmCalculatorActor::Construct()
{
  GateVActor::Construct();
//  Callbacks
//   EnableBeginOfRunAction(true);
//   EnableBeginOfEventAction(true);
//   EnablePreUserTrackingAction(true);
//   EnableUserSteppingAction(true);
//   ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin of Run
/*void GateEmCalculatorActor::BeginOfRunAction(const G4Run*r)
{
}*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Event
/*void GateEmCalculatorActor::BeginOfEventAction(const G4Event*e)
{
}*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Track
/*void GateEmCalculatorActor::PreUserTrackingAction(const GateVVolume * v, const G4Track*t)
{
}*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
/*void GateEmCalculatorActor::UserSteppingAction(const GateVVolume * v, const G4Step * step)
{
}*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateEmCalculatorActor::SaveData()
{
  GateVActor::SaveData();
  std::ofstream os;
  OpenFileOutput(mSaveFilename, os);

  double cut = DBL_MAX;
  double EmDEDX=0, NuclearDEDX=0, TotalDEDX=0;
  double density=0;
  double e=1.602176487e-19;
  double I=0;
  double eDensity=0;
  double radLength=0;
  G4double CrossSectionProcess = 0;
  G4double MuMassCoeficient = 0;
  G4String material;
  const G4MaterialTable* matTbl = G4Material::GetMaterialTable();

  G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
  G4ProcessVector* plist = particle->GetProcessManager()->GetProcessList();
  std::vector<G4String> processNameVector;
  for (G4int j = 0; j < plist->size(); j++)
    {
        if ( ( (*plist)[j]->GetProcessType() == fElectromagnetic) && ((*plist)[j]->GetProcessName() != "msc"))
            {
                processNameVector.push_back((*plist)[j]->GetProcessName());
            }
    }

      os << "# Output calculted for the following parameters:" << std::endl;
      os << "# Energy\t" << mEnergy << " MeV" << std::endl;
      os << "# Particle\t" << mPartName << "\n" << std::endl;
      os << "# And for the following materials" << std::endl;
// labels
      os << "Material\t";
      os << "Density\t\t";
      os << "e-density\t";
      os << "RadLength\t";
      os << "I\t";
      os << "EM-DEDX\t\t";
      os << "Nucl-DEDX\t";
      os << "Tot-DEDX\t";
      os << "Mu_mass" << std::endl;
// units
      os << "\t\t";
      os << "(g/cm³)\t\t";
      os << "(e-/mm³)\t";
      os << "(mm)\t\t";
      os << "(eV)\t";
      os << "(MeV.cm²/g)\t";
      os << "(MeV.cm²/g)\t";
      os << "(MeV.cm²/g)\t";
      os << "(cm²/g)" << std::endl;

  for(size_t k=0;k<G4Material::GetNumberOfMaterials();k++)
    {
      material = (*matTbl)[k]->GetName();
      density = (*matTbl)[k]->GetDensity();
      eDensity = (*matTbl)[k]->GetElectronDensity();
      radLength = (*matTbl)[k]->GetRadlen();
      I = (*matTbl)[k]->GetIonisation()->GetMeanExcitationEnergy();
      EmDEDX = emcalc->ComputeElectronicDEDX(mEnergy, mPartName, material, cut);
      NuclearDEDX = emcalc->ComputeNuclearDEDX(mEnergy, mPartName, material);
      TotalDEDX = emcalc->ComputeTotalDEDX(mEnergy, mPartName, material, cut);
      for( size_t j = 0; j < processNameVector.size(); j++)
        {
          CrossSectionProcess = emcalc->ComputeCrossSectionPerVolume( mEnergy, mPartName, processNameVector[j], material, cut);
          MuMassCoeficient += CrossSectionProcess / density;
        }


// Get methods issue
// for instance I tried:  double CSDARange = emcalc->GetDEDX(mEnergy, mPartName, material);
// I think geometries should be initialized first and then Get methods called and then physics and source could be initialized
// Currently all 3 initialization methods are called together, making difficult the use of GetMethods of G4EmCalculator.

// values
      os << material << "\t\t";
      os << density*e << "\t\t";
      os << eDensity << "\t";
      os << radLength << "\t\t";
      os << I*1.e6 << "\t";
      os << EmDEDX*10./(e*density) << "\t\t";
      os << NuclearDEDX*10./(e*density) << "\t";
      os << TotalDEDX*10./(e*density) << "\t\t";
      os << MuMassCoeficient << std::endl;
    }

  if (!os) {
    GateMessage("Output",1,"Error Writing file: " <<mSaveFilename << G4endl);
  }
  os.flush();
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEmCalculatorActor::ResetData()
{
}
//-----------------------------------------------------------------------------
