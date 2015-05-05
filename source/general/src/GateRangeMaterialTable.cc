#include "GateRangeMaterialTable.hh"
#include "GateDetectorConstruction.hh"

//-----------------------------------------------------------------------------
GateRangeMaterialTable::GateRangeMaterialTable()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateRangeMaterialTable::~GateRangeMaterialTable()
{
  for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mMaterialsVector.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterialDatabase(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  os << "[Materials]\n";
  for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); it++) 
  {
    os << "# Material corresponding to H=[ " 
       << it->mR1 << ";" << it->mR2 // << "],with density=[" 
      //        << G4BestUnit(md1[i],"Volumic Mass") 
      //        << ";" << G4BestUnit(md2[i],"Volumic Mass") << "]"
       << " ]\n";
    WriteMaterial(it->mMaterial, os);
    os << Gateendl;
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterialtoRangeLink(G4String filename) {
  std::ofstream os;
  OpenFileOutput(filename, os);
  for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); it++)
  {
    os << it->mR1 << " " << it->mR2 << " " << it->mMaterial->GetName() << Gateendl;    
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterial(G4Material * m, std::ofstream & os) {
  os << m->GetName() << ": d=" << G4BestUnit(m->GetDensity(),"Volumic Mass")
     << "; n=" << m->GetNumberOfElements() 
     << "; \n";
  for(unsigned int j=0; j<m->GetNumberOfElements(); j++) {
    os << "+el: name=" << m->GetElement(j)->GetName()
       << "; f=" << m->GetFractionVector()[j] << Gateendl;
  }
  //os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::AddMaterial(int R1, int R2, G4String name)
{
  //int n = GetNumberOfMaterials(); 
  //if(n==1) H1=mH2[n-1];
  mMaterials mat;
  mat.mR1 = R1;
  mat.mR2 = R2;
  mat.mName = name;
  mat.mMaterial = theMaterialDatabase.GetMaterial(name);
  mat.md1=mat.mMaterial->GetDensity();
  mMaterialsVector.push_back(mat);
  // Check
//  if (R2 < R1) GateError("R2=" << R2 << " is lower than R1=" << R1 << ". Abort.\n");
  //n++;

//  if (n != 1) {
//    if (R1 != mR2[n-2]) GateError("Current R1=" << R1 
//				  << " is different from previous R2=" 
//				  << mR2[n-2] << ". Abort.\n");
//  }
  
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::Reset()
{
  for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); )
  {
    it = mMaterialsVector.erase(it);
  }
  mMaterialsVector.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::MapLabelToMaterial(LabelToMaterialNameType & m)
{
  m.clear();int i = 0;
  for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin(); it != mMaterialsVector.end(); it++, i++) 
  {
    // GateMessage("Core", 0, 
    //               "i= " << i << " mi = "
    //             << m[i] << " mnamei = " 
    //              << mName[i] << Gateendl);
    std::pair<LabelType,G4String> lMaterial;
    lMaterial.first = i;
    lMaterial.second = it->mName;
    m.insert( lMaterial );
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateRangeMaterialTable::LabelType GateRangeMaterialTable::GetLabelFromR(int h)
{
  int i=0;
  while ((i<GetNumberOfMaterials() && h>=mMaterialsVector[i].mR1)) i++;
  i--;
  if ((i==GetNumberOfMaterials()-1) && h>mMaterialsVector[i].mR2) return i+1;
  return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateRangeMaterialTable::GetRMeanFromLabel(int l) {
  double h = (mMaterialsVector[l].mR1+mMaterialsVector[l].mR2)/2.0;
  return h;
}
//-----------------------------------------------------------------------------
