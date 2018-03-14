/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class  GateHounsfieldMaterialTable.hh
  \brief
  \author david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateHounsfieldMaterialTable__hh__
#define __GateHounsfieldMaterialTable__hh__

#include "GateHounsfieldMaterialProperties.hh"
#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"

class GateHounsfieldMaterialTable
{
public:
  GateHounsfieldMaterialTable();
  ~GateHounsfieldMaterialTable();
  typedef int LabelType;
  typedef std::map<LabelType,G4String> LabelToMaterialNameType;

  struct mMaterials
  {
    G4Material* mMaterial;
    double mH1;
    double mH2;
    double md1;
    G4String mName;
  };
  typedef std::vector<mMaterials> GateMaterialsVector;
  typedef GateMaterialsVector::iterator iterator;
  typedef GateMaterialsVector::const_iterator const_iterator;
  iterator begin(){ return mMaterialsVector.begin(); }
  iterator end(){ return mMaterialsVector.end(); }

  void AddMaterial(double H1, double H2, double d, GateHounsfieldMaterialProperties * p);
  void AddMaterial(double H1, double H2, G4String name);
  void WriteMaterialDatabase(G4String filename);
  void WriteMaterialtoHounsfieldLink(G4String filename);
  void WriteMaterial(G4Material * m, std::ofstream & os);
  int GetNumberOfMaterials() { return mMaterialsVector.size(); }
  void Reset();
  void MapLabelToMaterial(LabelToMaterialNameType & m);
  double GetHMeanFromLabel(int l);
  LabelType GetLabelFromH(double h);

  GateMaterialsVector GetMaterials() { return mMaterialsVector; }
  inline mMaterials & operator[](int index){ return mMaterialsVector[index];}

protected:
  GateMaterialsVector mMaterialsVector;

};
#endif
