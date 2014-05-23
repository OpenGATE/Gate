/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldMaterialProperties.cc
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#include "GateHounsfieldMaterialProperties.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialDatabase.hh"
#include "GateDetectorConstruction.hh"

//-----------------------------------------------------------------------------
GateHounsfieldMaterialProperties::GateHounsfieldMaterialProperties()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialProperties::~GateHounsfieldMaterialProperties()
{
  for (std::vector<G4Element*>::iterator it = mElementsList.begin(); it != mElementsList.end(); )
  {
    //delete (*it);
    it = mElementsList.erase(it);
  }

  mElementsFractionList.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateHounsfieldMaterialProperties::GetNumberOfElements()
{
  return mElementsList.size();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Element * GateHounsfieldMaterialProperties::GetElements(int i) 
{
  return mElementsList[i];
}
//-----------------------------------------------------------------------------

double GateHounsfieldMaterialProperties::GetElementsFraction(int i)
//-----------------------------------------------------------------------------
{
  return mElementsFractionList[i];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4String GateHounsfieldMaterialProperties::GetName() 
{
  return mName;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldMaterialProperties::GetH() 
{
  return mH;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialProperties::Read(std::ifstream & is, std::vector<G4String> & el) 
{
  skipComment(is);
  if (!is) return;
  double h;
  is >> h;  
  // Read/Create all elements fractions 
  for(unsigned int i=0; i<el.size(); i++) {
    ReadAndStoreElementFraction(is, el[i]);
    /*
    //  H    C    N    O   Na  Mg   P   S   Cl  Ar  K   Ca
    ReadAndStoreElementFraction(is, "Hydrogen");
    ReadAndStoreElementFraction(is, "Carbon");
    ReadAndStoreElementFraction(is, "Nitrogen");
    ReadAndStoreElementFraction(is, "Oxygen");
    ReadAndStoreElementFraction(is, "Sodium"); // Na
    ReadAndStoreElementFraction(is, "Magnesium");
    ReadAndStoreElementFraction(is, "Phosphor");
    ReadAndStoreElementFraction(is, "Sulfur");
    ReadAndStoreElementFraction(is, "Chlorine");
    ReadAndStoreElementFraction(is, "Argon");
    ReadAndStoreElementFraction(is, "Potassium");
    ReadAndStoreElementFraction(is, "Calcium");
    */
  }
  // Read name
  G4String n;
  is >> n; 
  // Set properties
  if (is) {
    mH= h;
    mName = n;
    // Normalise fraction
    double sum = 0.0;
    for(unsigned int i=0; i<mElementsFractionList.size(); i++) 
      sum += mElementsFractionList[i];
    for(unsigned int i=0; i<mElementsFractionList.size(); i++) 
      mElementsFractionList[i] /= sum;
    GateDebugMessage("Geometry", 6, "Read " << h << " " << mName << " " << sum << G4endl);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialProperties::ReadAndStoreElementFraction(std::ifstream & is, G4String name)
{
  skipComment(is);
  double f;
  is >> f;
  if (f>0) {
    if (is) {
      G4Element * e = theMaterialDatabase.GetElement(name);
      mElementsList.push_back(e);
      mElementsFractionList.push_back(f);
    }
    else {
      GateError("Error in reading element fraction of <" << name << ">" << G4endl);
    }
  }
}
//-----------------------------------------------------------------------------

