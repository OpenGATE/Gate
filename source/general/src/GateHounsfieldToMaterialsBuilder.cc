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
GateHounsfieldToMaterialsBuilder::GateHounsfieldToMaterialsBuilder()
: mDensityTol(0.1*g/cm3)// If user doesn't define tolerance assign a default value
{
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
  GateMessage("Geometry", 3, "GateHounsfieldToMaterialsBuilder::BuildAndWriteMaterials\n");

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
      GateError("The file " << mMaterialTableFilename << " must begin with [Elements]\n");
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
	      << ". Please check it.\n");
  }

  // Read densities.txt
  GateHounsfieldDensityTable * mDensityTable = new GateHounsfieldDensityTable();
  mDensityTable->Read(mDensityTableFilename);
  
  // Density tolerance
  double dTol = mDensityTol;

  // Declare result
  GateHounsfieldMaterialTable * mHounsfieldMaterialTable = new GateHounsfieldMaterialTable();

  // Loop on material intervals
  for(std::vector<GateHounsfieldMaterialProperties*>::iterator it=mHounsfieldMaterialPropertiesVector.begin();
		  it!=mHounsfieldMaterialPropertiesVector.end(); it++) {
	unsigned int i =   it-mHounsfieldMaterialPropertiesVector.begin();
    GateMessage("Geometry", 4, "Material " << i << " = " << (*it)->GetName() << Gateendl);
    
    double HMin = (*it)->GetH();
    double HMax;
    if (i == mHounsfieldMaterialPropertiesVector.size()-1) HMax = mDensityTable->GetHMax()+1;
    else {
    	HMax = (*(++it))->GetH();
    	it--;
    }
    
    // Check
    if (HMax <= HMin) GateError("Hounsfield shoud be given in ascending order, but I read H["
				<< i << "] = " << HMin
				<< " and H[" << i+1 << "] = " << HMax << Gateendl);
      GateMessage("Geometry", 4, "H " << HMin << " " << HMax << Gateendl);

    // Find densities interval (because densities not always increase)
    double dMin = mDensityTable->GetDensityFromH(HMin);
    double dMax = mDensityTable->GetDensityFromH(HMax);
     GateMessage("Geometry", 4, "Density " << G4BestUnit(dMin, "Volumic Mass") <<
    		 " " << G4BestUnit(dMax, "Volumic Mass") << Gateendl);
    double dDiffMax = mDensityTable->FindMaxDensityDifference(HMin, HMax);

    //if difference is small or material is air then split into 1 material only
    double n = (dDiffMax)/dTol; n = n<1 || (*it)->GetName()=="Air" ? 1 : n;
     GateMessage("Geometry", 4, "n = " << n << Gateendl);
    
    double HTol = (HMax-HMin)/n;
    GateMessage("Geometry", 4, "HTol = " << HTol << Gateendl);
    
    if (n>1) {
      GateMessage("Geometry", 4, "Material " << (*it)->GetName()
		  << " devided into " << n << " materials\n");
    }

    if (n<0) {
      GateError("ERROR Material " << (*it)->GetName()
		<< " devided into " << n << " materials : density decrease from " 
		<< G4BestUnit(dMin, "Volumic Mass") << " to " 
		<< G4BestUnit(dMax, "Volumic Mass") << Gateendl);
    }

    // Loop on density interval
    for(int j=0; j<n; j++) {
      double h1 = HMin+j*HTol;
      double h2 = std::min(HMin+(j+1)*HTol, HMax);
      //double d = mDensityTable->GetDensityFromH(h1+(h2-h1)/2.0);
      double d = mDensityTable->GetDensityFromH(h1);
        //density shall always be taken from the lower h, otherwise you will never have a real material description
       GateMessage("Geometry", 4, "H1/H2 " << h1 << " " << h2 << " = "
       		  << mHounsfieldMaterialPropertiesVector[i]->GetName()
       		  << " d=" << G4BestUnit(d, "Volumic Mass") << Gateendl);
      mHounsfieldMaterialTable->AddMaterial(h1, h2, d, *it);
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
	      << " materials.\n");
}
//-------------------------------------------------------------------------------------------------
