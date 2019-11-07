/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "G4SystemOfUnits.hh"
#include <G4NistManager.hh>

#include "GateMaterialDatabase.hh"
#include "GateMessageManager.hh"
#include "GateMDBFile.hh"

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL
#include "GateXMLDocument.hh"
#endif

//-----------------------------------------------------------------------------
GateMaterialDatabase::GateMaterialDatabase() {
  water_MPT = NULL;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialDatabase::~GateMaterialDatabase()
{
  if (water_MPT != NULL) delete water_MPT;
  std::vector<GateMDBFile*>::iterator i;
  for (i=mMDBFile.begin();i!=mMDBFile.end();++i) {
    delete (*i);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMaterialDatabase::AddMDBFile(const G4String& filename)
{
  mMDBFile.push_back(new GateMDBFile(this,filename));
  GateMessage("Materials",1, "New material database added - Number of files: "<<mMDBFile.size() << Gateendl);	  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Isotope* GateMaterialDatabase::GetIsotope(const G4String& isotopeName)
{
  GateMessage("Materials",5,"GateMaterialDatabase::GetIsotope("<<isotopeName<<")\n");
  G4Isotope* isotope = LookForIsotopeInTable(isotopeName);

  if (!isotope) {
    isotope = ReadIsotopeFromDBFile(isotopeName);
    if (!isotope)
      GateError("GateMaterialDatabase: failed to read the isotope '" << isotopeName << "' in the database file!");
  }
  return isotope;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
G4Element* GateMaterialDatabase::GetElement(const G4String& elementName)
{
  GateMessage("Materials",5,"GateMaterialDatabase::GetElement("<<elementName<<")\n");
  G4Element* element = LookForElementInTable(elementName);

  if (!element) {
    element = ReadElementFromDBFile(elementName);
    if (!element)
      GateError("GateMaterialDatabase: failed to read the element '" << elementName << "' in the database file!");
  }
  return element;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
G4Material* GateMaterialDatabase::GetMaterial(const G4String& materialName)
{
  GateMessage("Materials",3,"GateMaterialDatabase::GetMaterial("<<materialName<<")\n");
  
  G4Material* material = LookForMaterialInTable(materialName);
  
  if (!material) {
    material = ReadMaterialFromDBFile(materialName);
    if (!material)
      GateError("GateMaterialDatabase: failed to read the material '" << materialName << "' in the database file!" );
  }
  return material;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Isotope* GateMaterialDatabase::ReadIsotopeFromDBFile(const G4String& isotopeName)
{
  GateMessage("Materials",5,"GateMaterialDatabase::ReadIsotopeFromDBFile("<<isotopeName<<")\n");
  GateIsotopeCreator *CreatorTemp = 0;
  GateIsotopeCreator *Creator = 0;
  int nDef=0;
  G4String fileName= "";

  std::vector<GateMDBFile*>::iterator i;
  for (i=mMDBFile.begin();i!=mMDBFile.end();++i) {
    CreatorTemp = (*i)->ReadIsotope(isotopeName);
    if (CreatorTemp) {Creator = CreatorTemp; nDef++;fileName=(*i)->GetMDBFileName();}
    //if (Creator) break;
  }


  if (!Creator) GateError("GateMaterialDatabase: could not find the definition for isotope '" <<isotopeName << "' in material files");
  if(nDef>1) GateWarning("GateMaterialDatabase: Multiple definition of isotope: "<<isotopeName<<".\nThe definition in "<<fileName<<" was used.\n");
  G4Isotope* isotope =  Creator->Construct();
  delete Creator;
  return isotope;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Element* GateMaterialDatabase::ReadElementFromDBFile(const G4String& elementName)
{
  GateMessage("Materials",5,"GateMaterialDatabase::ReadElementFromDBFile("<<elementName<<")\n");
  GateElementCreator *CreatorTemp = 0;
  GateElementCreator *Creator = 0;
  int nDef=0;
  G4String fileName= "";
 
  std::vector<GateMDBFile*>::iterator i;
  for (i=mMDBFile.begin();i!=mMDBFile.end();++i) {
    CreatorTemp = (*i)->ReadElement(elementName);
    if (CreatorTemp) {Creator = CreatorTemp; nDef++;fileName=(*i)->GetMDBFileName();}
    //if (Creator) break;
  }


  if (!Creator) GateError("GateMaterialDatabase: could not find the definition for element '" <<elementName << "' in material files");
  if(nDef>1) GateWarning("GateMaterialDatabase: Multiple definition of element: "<<elementName<<".\nThe definition in "<<fileName<<" was used.\n");
  G4Element* element =  Creator->Construct();
  delete Creator;  
  return element;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Material* GateMaterialDatabase::ReadMaterialFromDBFile(const G4String& materialName)
{
  GateMessage("Materials",3,"GateMaterialDatabase::ReadMaterialFromDBFile("<<materialName<<")\n");
    
  GateMaterialCreator *Creator = 0;
  GateMaterialCreator *CreatorTemp = 0;
  G4Material* material = G4NistManager::Instance()->FindOrBuildMaterial(materialName);
  int nDef=0;
  G4String fileName= "";

  // If material is in NIST tables, you are done
  if(material!=NULL) return material;

  std::vector<GateMDBFile*>::iterator i;
  for (i=mMDBFile.begin();i!=mMDBFile.end();++i) {
    
    CreatorTemp = (*i)->ReadMaterial(materialName);
    if (CreatorTemp) {Creator = CreatorTemp; nDef++;fileName=(*i)->GetMDBFileName();}
    // if (Creator) break;
  }

  if (!Creator) {
    GateError("GateMaterialDatabase: could not find the definition for material '" << materialName << "' in material files");
    GateError("Check if you have defined a material database file in the macro file.");
  }

  if(nDef>1) GateWarning("GateMaterialDatabase: Multiple definition of material: "<<materialName<<".\nThe definition in "<<fileName<<" was used.\n");

  material = Creator->Construct();
  delete Creator;

  //------------------------------------------------------------------------------------------------
  if ((materialName == "Water") && (water_MPT == NULL)) {
    const G4int num_water=2;
    G4double pp_water[num_water] =
      { 2.034E-9*GeV, 4.136E-9*GeV };
    G4double rindex_water[num_water] =
      { 1., 1.};
    G4double abs_water[num_water] =
      {1000.*cm,  1000.*cm};
    water_MPT = new G4MaterialPropertiesTable();
    water_MPT->AddProperty("RINDEX",    pp_water, rindex_water, num_water);
    water_MPT->AddProperty("ABSLENGTH", pp_water, abs_water,    num_water);
    material->SetMaterialPropertiesTable(water_MPT);
  }
  //------------------------------------------------------------------------------------------------



  //------------------------------------------------------------------------------------------------
#ifdef GATE_USE_OPTICAL
  // initialize the properties table
  G4MaterialPropertiesTable* table = 0;
  // open the file
  GateXMLDocument* doc = new GateXMLDocument("./Materials.xml");
  // when the has opened correctly
  if (doc->Ok())
    {
      // find the material
      doc->Enter();
      if (doc->Find("material",materialName))
	{
	  // the material is found now read properties table
	  doc->Enter();
	  table = ReadMaterialPropertiesTable(doc);
	  doc->Leave();
	}
      // if the properties table is found, set it
      if (table) 
      { 
	  material->SetMaterialPropertiesTable(table);
	    
	      GateMessage("Materials",2, "GateMaterialDatabase: loaded material properties table for material '" + materialName + "'\n");
	      GateMessage("Materials",2,"  dumping properties table:\n");
	      GateMessage("Materials",2,"-----------------------------------------------------------------------------\n\n");
	      table->DumpTable();
	      GateMessage("Materials",2,"\n-----------------------------------------------------------------------------\n");
       }
       else
       {
	      GateMessage("Materials",2, "GateMaterialDatabase: did not load properties table for material '" + materialName + "'.\n");
	      GateMessage("Materials",2, "  This is only a problem when OPTICAL PHOTONS are transported in this material.'\n");
       }
    }
      // when openening of the xml-file failed
    else
    {
	  GateMessage("Materials",2, "GateMaterialDatabase: did not find the Materials.xml file: no properties read for material '" + materialName + "'\n");
	  GateMessage("Materials",2,"  This is only a problem when OPTICAL PHOTONS are transported in this material.'\n");
    }
     
#endif
      //------------------------------------------------------------------------------------------------


      return material;
    }
  //-----------------------------------------------------------------------------

