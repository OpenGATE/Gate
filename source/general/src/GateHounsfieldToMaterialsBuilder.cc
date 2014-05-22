/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*! 
  \class  GateHounsfieldToMaterialsBuilder.cc
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#include "GateHounsfieldToMaterialsBuilder.hh"
#include "GateHounsfieldMaterialTable.hh"
#include "GateHounsfieldDensityTable.hh"

//-------------------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilder::GateHounsfieldToMaterialsBuilder() {
  pMessenger = new GateHounsfieldToMaterialsBuilderMessenger(this);
  mMaterialTableFilename = "undefined_mMaterialTableFilename";
  mDensityTableFilename= "undefined_mDensityTableFilename";
  mOutputMaterialDatabaseFilename= "undefined_mOutputMaterialDatabaseFilename";
  mOutputHUMaterialFilename= "undefined_mOutputHUMaterialFilename";
}
//-------------------------------------------------------------------------------------------------
 

//-------------------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilder::~GateHounsfieldToMaterialsBuilder() {
  delete pMessenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateHounsfieldToMaterialsBuilder::BuildAndWriteMaterials() {
  GateMessage("Geometry", 3, "GateHounsfieldToMaterialsBuilder::BuildAndWriteMaterials" << G4endl);

  // Read matTable.txt
  std::vector<GateHounsfieldMaterialProperties*> mHounsfieldMaterialPropertiesVector;
  std::ifstream is;
  OpenFileInput(mMaterialTableFilename, is);
  skipComment(is);
  std::vector<G4String> elements;
  if (is) {
    G4String e;
    is >> e; 
    if (e != "[Elements]") {
      GateError("The file " << mMaterialTableFilename << " must begin with [Elements]" << G4endl);
    }
    while (e != "[/Elements]") {
      is >> e;
      if (e != "[/Elements]") elements.push_back(e);
    }
  }

  while (is) {
    GateHounsfieldMaterialProperties * p = new GateHounsfieldMaterialProperties();
    p->Read(is, elements);
    if (is) mHounsfieldMaterialPropertiesVector.push_back(p);
    else delete p;
  }
  
  //  DD(mHounsfieldMaterialPropertiesVector.size());

  if (mHounsfieldMaterialPropertiesVector.size() < 2) {
    GateError("I manage to read " << mHounsfieldMaterialPropertiesVector.size() 
	      << " materials in the file " << mMaterialTableFilename 
	      << ". Please check it." << G4endl);
  }

  // Read densities.txt
  GateHounsfieldDensityTable * mDensityTable = new GateHounsfieldDensityTable();
  mDensityTable->Read(mDensityTableFilename);
  
  // Density tolerance
  double dTol = mDensityTol;

  // Declare result
  GateHounsfieldMaterialTable * mHounsfieldMaterialTable = new GateHounsfieldMaterialTable();

  // Loop on material intervals
  for(unsigned int i=0; i<mHounsfieldMaterialPropertiesVector.size(); i++) {
    GateMessage("Geometry", 4, "Material " << i << " = " << mHounsfieldMaterialPropertiesVector[i]->GetName() << G4endl);
    
    double HMin = mHounsfieldMaterialPropertiesVector[i]->GetH();
    double HMax;
    if (i == mHounsfieldMaterialPropertiesVector.size()-1) HMax = HMin+1;
    else HMax = mHounsfieldMaterialPropertiesVector[i+1]->GetH();
    
    // Check
    if (HMax <= HMin) GateError("Hounsfield shoud be given in ascending order, but I read H["
				<< i << "] = " << HMin
				<< " and H[" << i+1 << "] = " << HMax << G4endl);
    // GateMessage("Core", 0, "H " << HMin << " " << HMax << G4endl);    

    // Find densities interval (because densities not always increase)
    double dMin = mDensityTable->GetDensityFromH(HMin);
    double dMax = mDensityTable->GetDensityFromH(HMax);
    // GateMessage("Core", 0, "Density " << dMin << " " << dMax << G4endl);    
    //     GateMessage("Core", 0, "Density " << dMin*g/cm3 << " " << dMax*g/cm3 << G4endl);   
    double dDiffMax = mDensityTable->FindMaxDensityDifference(HMin, HMax);

    double n = (dDiffMax)/dTol;
    // GateMessage("Core", 0, "n = " << n << G4endl);
    
    double HTol = (HMax-HMin)/n;
    // GateMessage("Core", 0, "HTol = " << HTol << G4endl);
    
    if (n>1) {
      GateMessage("Geometry", 4, "Material " << mHounsfieldMaterialPropertiesVector[i]->GetName() 
		  << " devided into " << n << " materials" << G4endl);
    }

    if (n<0) {
      GateError("ERROR Material " << mHounsfieldMaterialPropertiesVector[i]->GetName() 
		<< " devided into " << n << " materials : density decrease from " 
		<< G4BestUnit(dMin, "Volumic Mass") << " to " 
		<< G4BestUnit(dMax, "Volumic Mass") << G4endl);
    }

    // Loop on density interval
    for(int j=0; j<n; j++) {
      double h1 = HMin+j*HTol;
      double h2 = std::min(HMin+(j+1)*HTol, HMax);
      double d = mDensityTable->GetDensityFromH(h1+(h2-h1)/2.0);
      // GateMessage("Core", 0, "H1/H2 " << h1 << " " << h2 << " = " 
      // 		  << mHounsfieldMaterialPropertiesVector[i]->GetName() 
      // 		  << " d=" << G4BestUnit(d, "Volumic Mass") << G4endl);    
      mHounsfieldMaterialTable->AddMaterial(h1, h2, d, mHounsfieldMaterialPropertiesVector[i]);
    }
  }
  
  // Write final list of material
  mHounsfieldMaterialTable->WriteMaterialDatabase(mOutputMaterialDatabaseFilename);
  mHounsfieldMaterialTable->WriteMaterialtoHounsfieldLink(mOutputHUMaterialFilename);

  delete  mHounsfieldMaterialTable;
  delete mDensityTable;

  elements.clear();

  for (std::vector<GateHounsfieldMaterialProperties*>::iterator it = mHounsfieldMaterialPropertiesVector.begin();
                     it != mHounsfieldMaterialPropertiesVector.end(); )
  {
    delete (*it);
    it = mHounsfieldMaterialPropertiesVector.erase(it);
  }
  if(is) is.close();

  GateMessage("Geometry", 1, "Generation of " 
	      << mHounsfieldMaterialTable->GetNumberOfMaterials() 
	      << " materials." << G4endl);
}
//-------------------------------------------------------------------------------------------------
