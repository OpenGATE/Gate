#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"

class GateRangeMaterialTable
{
public:
  GateRangeMaterialTable();
  ~GateRangeMaterialTable();
  typedef int LabelType;
  typedef std::map<LabelType,G4String> LabelToMaterialNameType;

  struct mMaterials
  {
    G4Material* mMaterial;
    int mR1;
    int mR2;
    double md1;
    G4String mName;
  };
  typedef std::vector<mMaterials> GateMaterialsVector;
  typedef GateMaterialsVector::iterator iterator;
  typedef GateMaterialsVector::const_iterator const_iterator;
  iterator begin(){ return mMaterialsVector.begin(); }
  iterator end(){ return mMaterialsVector.end(); }
  
  void AddMaterial(int R1, int R2, G4String name); 
  void WriteMaterialDatabase(G4String filename);
  void WriteMaterialtoRangeLink(G4String filename);
  void WriteMaterial(G4Material * m, std::ofstream & os);
  int GetNumberOfMaterials() { return mMaterialsVector.size(); }
  void Reset();
  void MapLabelToMaterial(LabelToMaterialNameType & m);
  double GetRMeanFromLabel(int l);
  LabelType GetLabelFromR(int l);

  GateMaterialsVector GetMaterials() { return mMaterialsVector; }
  inline mMaterials & operator[](int index){ return mMaterialsVector[index];}

protected:
  GateMaterialsVector mMaterialsVector;
  
};

