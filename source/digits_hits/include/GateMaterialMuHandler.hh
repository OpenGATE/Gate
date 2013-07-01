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
  
  static GateMaterialMuHandler *GetInstance()
  {   
    if (singleton_MaterialMuHandler == 0)
    {
      singleton_MaterialMuHandler = new GateMaterialMuHandler();
    }
    return singleton_MaterialMuHandler;
  };
  
  
  ~GateMaterialMuHandler();
  double GetAttenuation(G4Material* material, double energy);
  double GetMu(G4Material* material, double energy);
  
private:
  
  GateMaterialMuHandler();  
  void AddMaterial(G4Material* material);
  void ReadElementFile(int z);
  void InitElementTable();
  void InitMaterialTable();
  
  map<G4String, GateMuTable*> mMaterialTable;
  GateMuTable** mElementsTable;
  int mNbOfElements;
  bool isInitialized;
  
  static GateMaterialMuHandler *singleton_MaterialMuHandler;
  
};


#endif
