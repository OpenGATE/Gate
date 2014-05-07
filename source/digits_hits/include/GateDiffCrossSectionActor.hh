/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateDiffCrossSectionActor
  \author edward.romero@creatis.insa-lyon.fr
 */

#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

#include "GateConfiguration.h"
#ifndef GateDiffCrossSectionActor_HH
#define GateDiffCrossSectionActor_HH
#include "G4UnitsTable.hh"
#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateDiffCrossSectionActorMessenger.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4EmCalculator.hh"
#include "G4Electron.hh"
#include <list>

class G4VEMDataSet;
class GateDiffCrossSectionActor : public GateVActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateDiffCrossSectionActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateDiffCrossSectionActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run* run);

 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialise();
  void SetEnergy( G4double value);
  void ReadListEnergy( G4String filename);

  void SetAngle( G4double value);
  void ReadListAngle( G4String filename);

  void SetMaterial(G4String name);
  void ReadMaterialList(G4String filename);

  ofstream PointerToFileDataOutSF;
  ofstream PointerToFileDataOutFF;
  ofstream PointerToFileDataOutDCScompton;
  ofstream PointerToFileDataOutDCSrayleigh;
  std::stringstream DriverDataOutSF;
  std::stringstream DriverDataOutFF;
  std::stringstream DriverDataOutDCScompton;
  std::stringstream DriverDataOutDCSrayleigh;
  std::string DataOutSF;
  std::string DataOutFF;
  std::string DataOutDCScompton;
  std::string DataOutDCSrayleigh;
protected:
  GateDiffCrossSectionActor(G4String name, G4int depth=0);
  GateDiffCrossSectionActorMessenger * pMessenger;
  G4double mUserEnergy;
  G4double mUserAngle;
  G4String mUserMaterial;
  G4String mExitFileNameSF, mExitFileNameFF, mExitFileNameDCScompton, mExitFileNameDCSrayleigh;
  std::vector<G4double > mUserEnergyList;
  std::vector<G4double > mUserAngleList;
  std::vector<G4String > mUserMaterialList;
  G4VEMDataSet* scatterFunctionData;
  G4VEMDataSet* formFactorData;

};

MAKE_AUTO_CREATOR_ACTOR(DiffCrossSectionActor,GateDiffCrossSectionActor)

#endif
