/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateMaterialMuHandler
  \author fabien.baldacci@creatis.insa-lyon.fr
 */

#ifndef GATEMATERIALMUHANDLER_HH
#define GATEMATERIALMUHANDLER_HH

#include "G4UnitsTable.hh"
#include "GateMuTables.hh"
#include "G4Material.hh"
#include <map>
using std::map;
using std::string;

class GateMaterialMuHandler
{

public:
  GateMaterialMuHandler(int nbOfElements);
  ~GateMaterialMuHandler();
  void AddMaterial(G4Material* material);
  double GetAttenuation(G4Material* material, double energy);
  double GetMu(G4Material* material, double energy);

  void InitMaterialTable();
  
private:
  map<G4String, GateMuTable*> mMaterialTable;
  GateMuTable** mElementsTable;
  int mNbOfElements;
  
  void ReadElementFile(int z);
  void InitElementTable();
  
};


#endif
