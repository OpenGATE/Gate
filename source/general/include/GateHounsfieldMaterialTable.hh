/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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

  void AddMaterial(double H1, double H2, double d, GateHounsfieldMaterialProperties * p);
  void AddMaterial(double H1, double H2, G4String name); 
  void WriteMaterialDatabase(G4String filename);
  void WriteMaterialtoHounsfieldLink(G4String filename);
  void WriteMaterial(G4Material * m, std::ofstream & os);
  int GetNumberOfMaterials() { return mH1.size(); }
  void Reset();
  void MapLabelToMaterial(LabelToMaterialNameType & m);
  double GetHMeanFromLabel(int l);
  LabelType GetLabelFromH(double h);
  std::vector<double> & GetH1Vector() { return mH1; }
  std::vector<double> & GetH2Vector() { return mH2; }
  std::vector<double> & GetDVector() { return md1; }

protected:
  std::vector<G4Material*> mMaterialsVector;
  std::vector<double> mH1;
  std::vector<double> mH2;
  std::vector<double> md1;
  // std::vector<double> md2;
  std::vector<G4String> mName;

};
#endif
