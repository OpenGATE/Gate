#include "GateRangeMaterialTable.hh"

//-----------------------------------------------------------------------------
GateRangeMaterialTable::GateRangeMaterialTable()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateRangeMaterialTable::~GateRangeMaterialTable()
{
  for (std::vector<G4Material*>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mR1.clear();
  mR2.clear();
  md1.clear();
  mName.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterialDatabase(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  os << "[Materials]" << G4endl;
  for(unsigned int i=0; i<mMaterialsVector.size(); i++) {
    os << "# Material " << i << " corresponding to H=[ " 
       << mR1[i] << ";" << mR2[i] // << "],with density=[" 
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
void GateRangeMaterialTable::WriteMaterialtoRangeLink(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  for(unsigned int i=0; i<mMaterialsVector.size(); i++) {
    os << mR1[i] << " " << mR2[i] << " " << mMaterialsVector[i]->GetName() << G4endl;    
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterial(G4Material * m, std::ofstream & os) {
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
void GateRangeMaterialTable::AddMaterial(int R1, int R2, G4String name)
{
  int n = GetNumberOfMaterials(); 
  //if(n==1) H1=mH2[n-1];

  mR1.push_back(R1);
  mR2.push_back(R2);
  mName.push_back(name);
  // Check
//  if (R2 < R1) GateError("R2=" << R2 << " is lower than R1=" << R1 << ". Abort." << G4endl);
  n++;

//  if (n != 1) {
//    if (R1 != mR2[n-2]) GateError("Current R1=" << R1 
//				  << " is different from previous R2=" 
//				  << mR2[n-2] << ". Abort." << G4endl);
//  }
  
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::Reset()
{
  for (std::vector<G4Material*>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mMaterialsVector.clear();
  mR1.clear();
  mR2.clear();
  md1.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::MapLabelToMaterial(LabelToMaterialNameType & m)
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
GateRangeMaterialTable::LabelType GateRangeMaterialTable::GetLabelFromR(int h)
{
  int i=0;
  while ((i<GetNumberOfMaterials() && h>=mR1[i])) i++;
  i--;
  if ((i==GetNumberOfMaterials()-1) && h>mR2[i]) return i+1;
  return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateRangeMaterialTable::GetRMeanFromLabel(int l) {
  double h = (mR1[l]+mR2[l])/2.0;
  return h;
}
//-----------------------------------------------------------------------------
