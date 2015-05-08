#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"
#include "G4VisAttributes.hh"

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
    G4VisAttributes* mVisAttributes;
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
  
  void AddMaterial(int R1, int R2, G4String name, G4bool visibility=true, G4Colour Color=G4Colour(0.5,0.,0.,1.));
  void AddMaterial(int R1, int R2, G4String name, G4VisAttributes* Attributes);
  void WriteMaterialDatabase(G4String filename);
  void WriteMaterialtoRangeLink(G4String filename);
  void WriteMaterial(G4Material * m, std::ofstream & os);
  int GetNumberOfMaterials() { return mMaterialsVector.size(); }
  void Reset();
  void MapLabelToMaterial(LabelToMaterialNameType & m);
  double GetRMeanFromLabel(int l);
  LabelType GetLabelFromR(int l);
  inline void push_back(mMaterials MM){ mMaterialsVector.push_back(MM); }
  inline void pop_back(){ mMaterialsVector.pop_back(); }
  inline void clear(){ mMaterialsVector.clear(); }

  GateMaterialsVector GetMaterials() { return mMaterialsVector; }
  inline mMaterials & operator[](int index){ return mMaterialsVector[index];}

protected:
  GateMaterialsVector mMaterialsVector;
  
};

