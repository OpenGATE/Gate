/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



#ifndef GateMDBFile_hh
#define GateMDBFile_hh

#include "globals.hh"
#include <fstream>

#include "G4Material.hh"

#include "GateMDBCreators.hh"
#include "GateMDBFieldReader.hh"

class GateMaterialDatabase;

class GateMDBFile : public GateMDBFieldReader
{
public:
  /// Constructor. Takes the Database which uses this and the filename of the part of the database it is responsible to read
  GateMDBFile(GateMaterialDatabase* db, const G4String& itsFileName);
  virtual ~GateMDBFile();

public:
  GateIsotopeCreator*  ReadIsotope(const G4String& isotopeName);
  GateElementCreator*  ReadElement(const G4String& elementName);
  GateMaterialCreator* ReadMaterial(const G4String& materialName);
  G4String GetMDBFileName(){return filePath;}

protected:
  GateScratchElementCreator*  ReadScratchElement(const G4String& elementName,const G4String& line);
  GateCompoundElementCreator* ReadCompoundElement(const G4String& elementName,const G4String& line);

  GateScratchMaterialCreator*  ReadScratchMaterial(const G4String& materialName,const G4String& line);
  GateCompoundMaterialCreator* ReadCompoundMaterial(const G4String& materialName,const G4String& line);

  GateComponentCreator*      ReadComponent(const G4String& materialName,const G4String& componentOrdinal);
  GateElemComponentCreator*  ReadElemComponent(const G4String& materialName,const G4String& componentOrdinal,const G4String& line);
  GateEByNComponentCreator*  ReadEByNComponent(const G4String& materialName,const G4String& componentOrdinal,
      	      	      	      	      	       const G4String& field, const G4String& componentName);
  GateEByFComponentCreator*  ReadEByFComponent(const G4String& materialName,const G4String& componentOrdinal,
      	      	      	      	      	       const G4String& field, const G4String& componentName);
  GateIByFComponentCreator*  ReadIsoComponent(const G4String& elementName,const G4String& componentOrdinal,const G4String& line);
  GateMatComponentCreator*   ReadMatComponent(const G4String& materialName,const G4String& componentOrdinal,const G4String& line);
  
  void     ReadAllMaterialOptions(const G4String& materialName,const G4String& line,GateMaterialCreator* creator);
  void     ReadMaterialOption(const G4String& materialName,const G4String& field,GateMaterialCreator* creator);

  void     FindSection(const G4String& name);
  G4bool   LookForText(const G4String& text);
  G4String ReadItem(const G4String& sectionName,const G4String& itemName);
  G4int    ReadNonEmptyLine(G4String& lineBuffer);
  G4int    ReadLine(G4String& lineBuffer);

private:
  // Stores the database which instanciated this (used by creators)
  GateMaterialDatabase* mDatabase;
  G4String fileName;
  G4String filePath;
  std::ifstream dbStream;

public:
  static char theStarterSeparator;
  static char theFieldSeparator;
  static G4String theReadItemErrorMsg;
};

#endif
