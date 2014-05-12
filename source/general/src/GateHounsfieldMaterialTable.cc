/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*! 
  \class  GateHounsfieldMaterialTable.cc
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#include "GateHounsfieldMaterialTable.hh"
#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::GateHounsfieldMaterialTable()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::~GateHounsfieldMaterialTable()
{
  for (std::vector<G4Material*>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mH1.clear();
  mH2.clear();
  md1.clear();
  mName.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::AddMaterial(double H1, double H2, double d, 
					      GateHounsfieldMaterialProperties * p) {
    
  // Check
  if (H1>=H2) 
    GateError("Error in GateHounsfieldMaterialProperties::AddMaterial " << H1 << " " << H2 << G4endl);
    
  // Set values
  mH1.push_back(H1);
  mH2.push_back(H2);
  md1.push_back(d);

  // Material's name
  G4String name = p->GetName()+"_"+DoubletoString(mMaterialsVector.size());

  // Create new material
  G4Material * m = new G4Material(name, d, p->GetNumberOfElements());

  // Material's elements
  for(int i=0; i<p->GetNumberOfElements(); i++) {
    m->AddElement(p->GetElements(i), p->GetElementsFraction(i));
  }

  // Set material
  mMaterialsVector.push_back(m);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterialDatabase(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  os << "[Materials]" << G4endl;
  for(unsigned int i=0; i<mMaterialsVector.size(); i++) {
    os << "# Material " << i << " corresponding to H=[ " 
       << mH1[i] << ";" << mH2[i] // << "],with density=[" 
      //        << G4BestUnit(md1[i],"Volumic Mass") 
      //        << ";" << G4BestUnit(md2[i],"Volumic Mass") << "]"
       << " ]" << G4endl;
    WriteMaterial(mMaterialsVector[i], os);
    os << G4endl;
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterialtoHounsfieldLink(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  for(unsigned int i=0; i<mMaterialsVector.size(); i++) {
    os << mH1[i] << " " << mH2[i] << " " << mMaterialsVector[i]->GetName() << G4endl;    
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterial(G4Material * m, std::ofstream & os) {
  os << m->GetName() << ": d=" << G4BestUnit(m->GetDensity(),"Volumic Mass")
     << "; n=" << m->GetNumberOfElements() 
     << "; " << std::endl;
  for(unsigned int j=0; j<m->GetNumberOfElements(); j++) {
    os << "+el: name=" << m->GetElement(j)->GetName()
       << "; f=" << m->GetFractionVector()[j] << std::endl;
  }
  //os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::AddMaterial(double H1, double H2, G4String name)
{
  int n = GetNumberOfMaterials(); 
  //if(n==1) H1=mH2[n-1];

  mH1.push_back(H1);
  mH2.push_back(H2);
  mName.push_back(name);
  // Check
  if (H2 < H1) GateError("H2=" << H2 << " is lower than H1=" << H1 << ". Abort." << G4endl);
  n++;

  if (n != 1) {
    if (H1 != mH2[n-2]) GateError("Current H1=" << H1 
				  << " is different from previous H2=" 
				  << mH2[n-2] << ". Abort." << G4endl);
  }
  
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::Reset()
{
  for (std::vector<G4Material*>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mMaterialsVector.clear();
  mH1.clear();
  mH2.clear();
  md1.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::MapLabelToMaterial(LabelToMaterialNameType & m)
{
  m.clear();
  for(int i=0; i<GetNumberOfMaterials(); i++) {
    // GateMessage("Core", 0, 
    //               "i= " << i << " mi = "
    //             << m[i] << " mnamei = " 
    //              << mName[i] << G4endl);
    m[i] = mName[i];
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::LabelType GateHounsfieldMaterialTable::GetLabelFromH(double h)
{
  int i=0;
  while ((i<GetNumberOfMaterials() && h>=mH1[i])) i++;
  i--;
  if ((i==GetNumberOfMaterials()-1) && h>mH2[i]) return i+1;
  return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldMaterialTable::GetHMeanFromLabel(int l) {
  double h = (mH1[l]+mH2[l])/2.0;
  return h;
}
//-----------------------------------------------------------------------------
