/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class GateMaterialDatabase
  \ingroup scene
*/

#ifndef GateMaterialDatabase_hh
#define GateMaterialDatabase_hh

#include "globals.hh"
#include <fstream>
#include <vector>
#include "GateMDBCreators.hh"

class GateMDBFile;

class G4Element;
class G4Material;

//#define DEFAULT_GATESURFACEMATDB "Materials.xml"

class GateMaterialDatabase
{
public:
  GateMaterialDatabase();
  ~GateMaterialDatabase();

  void AddMDBFile(const G4String& filename);

  G4Element*  GetElement(const G4String& name);
  G4Material* GetMaterial(const G4String& materialName);


protected:
  G4Element*  ReadElementFromDBFile(const G4String& elementName)  ;
  G4Material* ReadMaterialFromDBFile(const G4String& materialName);

  //'false' to avoid warning messages!
  inline G4Element*  LookForElementInTable(const G4String& elementName)   { return G4Element::GetElement(elementName,false); }
  inline G4Material* LookForMaterialInTable(const G4String& materialName) { return G4Material::GetMaterial(materialName, false); }

private:
  std::vector<GateMDBFile*> mMDBFile;
  G4MaterialPropertiesTable * water_MPT;
};

#endif
