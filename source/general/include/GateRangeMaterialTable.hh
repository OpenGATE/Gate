#include "GateMiscFunctions.hh"

class GateRangeMaterialTable
{
public:
  GateRangeMaterialTable();
  ~GateRangeMaterialTable();
  typedef int LabelType;
  typedef std::map<LabelType,G4String> LabelToMaterialNameType;

  void AddMaterial(int R1, int R2, G4String name); 
  void WriteMaterialDatabase(G4String filename);
  void WriteMaterialtoRangeLink(G4String filename);
  void WriteMaterial(G4Material * m, std::ofstream & os);
  int GetNumberOfMaterials() { return mR1.size(); }
  void Reset();
  void MapLabelToMaterial(LabelToMaterialNameType & m);
  double GetRMeanFromLabel(int l);
  LabelType GetLabelFromR(int l);
  std::vector<int> & GetR1Vector() { return mR1; }
  std::vector<int> & GetR2Vector() { return mR2; }
  std::vector<int> & GetDVector() { return md1; }

protected:
  std::vector<G4Material*> mMaterialsVector;
  std::vector<int> mR1;
  std::vector<int> mR2;
  std::vector<int> md1;
  // std::vector<double> md2;
  std::vector<G4String> mName;

};

