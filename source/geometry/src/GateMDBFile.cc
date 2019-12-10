/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/



/*!

  \file GateMDBFile.cc

  \brief Class GateMDBFile
*/

#include "GateMDBFile.hh"
#include "GateMDBCreators.hh"

#include "GateMaterialDatabase.hh"
#include "GateMessageManager.hh"

#include "GateTokenizer.hh"
#include "GateTools.hh"

char GateMDBFile::theStarterSeparator = ':';
char GateMDBFile::theFieldSeparator   = ';';
G4String GateMDBFile::theReadItemErrorMsg = "Item not found";

#define GATE_BUFFERSIZE 256

//-----------------------------------------------------------------------------
GateMDBFile::GateMDBFile(GateMaterialDatabase* db, const G4String& itsFileName)
  :mDatabase(db), 
   fileName(itsFileName),filePath("")
{
  GateMessage("Materials", 1, 
	      "GateMDBFile: I start looking for the material database file <"
	      << fileName << ">\n"); 
  filePath = GateTools::FindGateFile(fileName);
  if (filePath.empty())
	{
		G4String msg = "Could not find material database file '" + fileName + "'";
    G4Exception( "GateMDBFile::GateMDBFile", "GateMDBFile", FatalException, msg );
	}
  dbStream.open(filePath);

  if (dbStream) {
    GateMessage("Materials", 2, 
		"OK, I opened the material database <" 
		<< filePath << ">\n");
  }
  else {
		G4String msg = "Could not open material database file '" + filePath + "'";
    G4Exception( "GateMDBFile::GateMDBFile", "GateMDBFile", FatalException, msg );
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
GateMDBFile::~GateMDBFile()
{
  dbStream.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Read a new isotope from the DB file --> returns an isotope creator
GateIsotopeCreator* GateMDBFile::ReadIsotope(const G4String& isotopeName)
{
  GateMessage("Materials", 5,
	      "GateMDBFile<" << fileName
	      << ">::ReadIsotope("
	      << isotopeName <<")\n");

  // Find the isotope definition line in the [Isotopes] section of the DB file
  G4String line = ReadItem("Isotopes", isotopeName);
  if (line == theReadItemErrorMsg)  return 0;

  GateMessage("Materials", 5,
	      "GateMDBFile<" << fileName
	      << ">: found definition for isotope '"
	      << isotopeName << "' as an elementary isotope.\n");

  if (line == "")
    DecodingException(isotopeName,"\tThe isotope's definition line seems to be empty\n");

  // Create an empty isotope-creator
  GateIsotopeCreator *creator = new GateIsotopeCreator(isotopeName);

  // Read the 1st field as the isotope's atomic number (Z)
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  creator->atomicNumber = ReadAtomicNumber(isotopeName,stringPair.first);

  // Read the 2nd field as the element's nucleon number (N)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  creator->nucleonNumber = ReadNucleonNumber(isotopeName,stringPair.first);

  // Read the 3rd field as the element's molar mass (A)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  creator->molarMass = ReadMolarMass(isotopeName,stringPair.first);

  GateMessage("Materials", 5,
	      "GateMDBFile<" << fileName
	      << ": definition loaded for isotope '"
	      << isotopeName <<"'.\n");

  return creator;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read a new element from the DB file --> returns an element creator
GateElementCreator* GateMDBFile::ReadElement(const G4String& elementName)
{
  GateMessage("Materials", 5, 
	      "GateMDBFile<" << fileName 
	      << ">::ReadElement(" 
	      << elementName <<")\n");

  // Find the element definition line in the [Elements] section of the DB file
  G4String line = ReadItem("Elements",elementName);
  if (line == theReadItemErrorMsg)  return 0;

  GateMessage("Materials", 5,  
	      "GateMDBFile<" << fileName
	      << ">: found definition for element '" 
	      << elementName << "' as an elementary element.\n");

  if (line == "") 
    DecodingException(elementName,"\tThe element's definition line seems to be empty\n");
  
  // Check the content of the first field to decide what kind of element it will be (scratch or compound)
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  ElementType type = EvaluateElementType(elementName,stringPair.first);

  // Launch the appropriate element readout function
  switch (type) {
  case elementtype_scratch:
    return ReadScratchElement(elementName,line);
  case elementtype_compound:
    return ReadCompoundElement(elementName,line);
  default:
		G4String msg = "Abnormal prefix code found for the first field of element '" + elementName + "'";
    G4Exception( "GateMDBFile::ReadElement", "ReadElement", FatalException, msg );
  }
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateScratchElementCreator*  GateMDBFile::ReadScratchElement(const G4String& elementName,const G4String& line)
{
  // Create an empty element-creator
  GateScratchElementCreator *creator = new GateScratchElementCreator(elementName);
  
  // Read the 1st field as the element's symbol
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);  
  creator->symbol = ReadElementSymbol(elementName,stringPair.first);

  // Read the 2nd field as the element's atomic number (Z)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);  
  creator->atomicNumber = ReadAtomicNumber(elementName,stringPair.first);

  // Read the 3rd field as the element's molar mass (A)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);  
  creator->molarMass = ReadMolarMass(elementName,stringPair.first);

  GateMessage("Materials", 5, 
	      "GateMDBFile<" << fileName
	      << ": definition loaded for element '" 
	      << elementName <<"'.\n");
  return creator;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read a new compound element from the DB file --> returns a element creator
GateCompoundElementCreator* GateMDBFile::ReadCompoundElement(const G4String& elementName,const G4String& line)
{
  GateMessage("Elements", 5,  "GateMDBFile<" << fileName
	      <<">: found definition for element '"
	      << elementName << "' as a compound element.\n");

  // Create an empty element-creator
  GateCompoundElementCreator *creator = new GateCompoundElementCreator(elementName);

  // Read the 1st field as the number of components
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  creator->nComponents = ReadNumberOfComponents(elementName,stringPair.first);

  // Read the 2nd field as the symbol
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  creator->symbol = ReadElementSymbol(elementName,stringPair.first);

  GateMessage("Elements", 5,
	      "GateMDBFile: element '"
	      << elementName << "' has "
	      << creator->nComponents << " components.\n");

  // Read and store all element's components
  for (G4int i=1;i<=creator->nComponents;i++) {
    G4String componentOrdinal = CreateOrdinalString(i);
    GateComponentCreator* componentCreator = ReadComponent(elementName,componentOrdinal);
    creator->components.push_back(componentCreator);
  }

  GateMessage("Elements", 5,
	      "GateMDBFile: definition loaded for material '"
	      << elementName << "'.\n");

  return creator;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read a new material from the DB file --> returns a material creator
GateMaterialCreator* GateMDBFile::ReadMaterial(const G4String& materialName)
{  
  GateMessage("Materials", 3, 
	      "GateMDBFile<" << fileName 
	      << ">::ReadMaterial(" << materialName<<")\n");

  // Find the material definition line in the [Materials] section of the DB file
  G4String line = ReadItem("Materials",materialName);
  if (line == theReadItemErrorMsg) {
    //GateMessage("Materials", 4, "GateMDBFile<"<<fileName<< ">::ReadMaterial("<< materialName<<") ReadError\n");
    return 0;
  }

  GateMessage("Materials", 4, "GateMDBFile<"<<fileName<< ">::ReadMaterial("<< materialName<<") found. \n");

  if (line == "") 
    DecodingException(materialName,"\tThe material's definition line seems to be empty\n");

  // Check the content of the first field to decide what kind of material it will be (scratch or compound)
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);  
  MaterialType type = EvaluateMaterialType(materialName,stringPair.first);

  // Launch the appropriate material readout function
  switch (type) {
  case materialtype_scratch:
    return ReadScratchMaterial(materialName,line);
  case materialtype_compound:
    return ReadCompoundMaterial(materialName,line);
  default:
		G4String msg = "Abnormal prefix code found for the first field of material '" + materialName + "'";
    G4Exception( "GateMDBFile::ReadMaterial", "ReadMaterial", FatalException, msg );
  }
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new material to create from scratch from the DB file --> returns a material creator
GateScratchMaterialCreator* GateMDBFile::ReadScratchMaterial(const G4String& materialName,const G4String& line)
{
  GateMessage("Materials", 3, "GateMDBFile<" << fileName
	      <<">: found definition for material '" 
	      << materialName << "' as an elementary material.\n\n");
  
  // Create an empty material-creator
  GateScratchMaterialCreator *creator = new GateScratchMaterialCreator(materialName);

  // Read the 1st field as the atomic number (Z)
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);  
  creator->atomicNumber = ReadAtomicNumber(materialName,stringPair.first);

  // Read the 2nd field as the material's molar mass (A)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);  
  creator->molarMass = ReadMolarMass(materialName,stringPair.first);

  // Read the 3rd field as the density
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);  
  creator->density = ReadDensity(materialName,stringPair.first);

  // Process all other remaining fields (if any) to read the option data (state, density, presure)
  ReadAllMaterialOptions(materialName,stringPair.second,creator);

  GateMessage("Materials", 3, 
	      "GateMDBFile: definition loaded for material '" 
	      << materialName << "'.\n\n");
  return creator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new compound material from the DB file --> returns a material creator
GateCompoundMaterialCreator* GateMDBFile::ReadCompoundMaterial(const G4String& materialName,const G4String& line)
{
  GateMessage("Materials", 5,  "GateMDBFile<" << fileName
	      <<">: found definition for material '" 
	      << materialName << "' as a compound material.\n");
  
  // Create an empty material-creator
  GateCompoundMaterialCreator *creator = new GateCompoundMaterialCreator(materialName);

  // Read the 1st field as the density
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);  
  creator->density = ReadDensity(materialName,stringPair.first);

  // Read the 2nd field as the number of components
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);  
  creator->nComponents = ReadNumberOfComponents(materialName,stringPair.first);

  GateMessage("Materials", 5, 
	      "GateMDBFile: material '" 
	      << materialName << "' has " 
	      << creator->nComponents << " components.\n");

  // Process all other remaining fields (if any) to read the option data (state, density, pressure)
  ReadAllMaterialOptions(materialName,stringPair.second,creator);

  // Read and store all material's components
  for (G4int i=1;i<=creator->nComponents;i++) {
    G4String componentOrdinal = CreateOrdinalString(i);
    GateComponentCreator* componentCreator = ReadComponent(materialName,componentOrdinal);
    creator->components.push_back(componentCreator);
  }

  GateMessage("Materials", 5, 
	      "GateMDBFile: definition loaded for material '" 
	      << materialName << "'.\n");

  return creator;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read a new material component for the current compound material
GateComponentCreator* GateMDBFile::ReadComponent(const G4String& materialName,const G4String& componentOrdinal)
{
  // Read the next non-empty line (if any)
  G4String line;
  if (ReadNonEmptyLine(line)) {
		G4String msg = "I could not find the " +  componentOrdinal +  " component of the compound material '" + materialName + "' ";
		msg += "You should check the list of components in the database file for this material.";
    G4Exception( "GateMDBFile::ReadComponent", "ReadComponent", FatalException, msg );
  }

  // Check the line starter to see whether it's a material-type or an element-type component
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theStarterSeparator);  
  ComponentType type = EvaluateComponentType(materialName,stringPair.first,componentOrdinal);

  // Launch the appropriate component reader
  switch (type) {
  case componenttype_elem:
    return ReadElemComponent(materialName,componentOrdinal,stringPair.second);
    break;
  case componenttype_mat:
    return ReadMatComponent(materialName,componentOrdinal,stringPair.second);
    break;
  case componenttype_iso:
    return ReadIsoComponent(materialName,componentOrdinal,stringPair.second);
    break;
  default:
		G4String msg = "Incorrect definition line for the  " +  componentOrdinal +  " component of the compound material '" + materialName + "'";
		msg += " This line should start with '+el:', '+mat:' or '+iso:'. You should check the list of components in the database file for this material.";
    G4Exception("GateMDBFile::ReadComponent", "ReadComponent" , FatalException, msg );
  }
  return 0;
}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
// Read a new element-type material component for the current compound material
GateElemComponentCreator* GateMDBFile::ReadElemComponent(const G4String& materialName,const G4String& componentOrdinal,const G4String& line)
{
  // Reads the 1st field (after the starter) as the component's name
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  G4String name = ReadComponentName(materialName,componentOrdinal,stringPair.first);
  if ( name == "auto" )
    name = materialName;
  
  // Chack the component's abundance field
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  G4String field=stringPair.first;
  ElemComponentType type = EvaluateElemComponentType(materialName,field,componentOrdinal);

  switch (type) {
  case elemcomponent_byNAtoms:
    return ReadEByNComponent(materialName,componentOrdinal,field,name);
  case elemcomponent_byFraction:
    return ReadEByFComponent(materialName,componentOrdinal,field,name);
  default:
		G4String msg = "Abnormal prefix code found for the abundance field of  "+ componentOrdinal + " component of material '" + materialName + "'";
    G4Exception( "GateMDBFile::ReadElemComponent", "ReadElemComponent", FatalException, msg );
  }
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new element-type material component for the current compound material with the abundance passe by NAtoms
GateEByNComponentCreator* GateMDBFile::ReadEByNComponent(const G4String& materialName,const G4String& componentOrdinal,
							 const G4String& field, const G4String& componentName)
{
  GateEByNComponentCreator *creator = new GateEByNComponentCreator(mDatabase,componentName);
  creator->nAtoms = ReadComponentNAtoms(materialName,componentOrdinal,field);
  GateMessage("Materials", 5,  
	      "GateMDBFile: " << componentOrdinal 
	      << " component is the element '" << creator->name
	      << "' (number of atoms= " << creator->nAtoms << ")\n");
  
  return creator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new element-type material component for the current compound material with the abundance passe by fraction
GateEByFComponentCreator* GateMDBFile::ReadEByFComponent(const G4String& materialName,const G4String& componentOrdinal,
							 const G4String& field, const G4String& componentName)
{
  GateEByFComponentCreator *creator = new GateEByFComponentCreator(mDatabase,componentName);
  creator->fraction = ReadComponentFraction(materialName,componentOrdinal,field);
  GateMessage("Materials", 5,  
	      "GateMDBFile: " << componentOrdinal 
	      << " component is the element '" << creator->name
	      << "' (fraction= " << creator->fraction << ")\n");
  
  return creator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new material-type material component for the current compound material
GateMatComponentCreator* GateMDBFile::ReadMatComponent(const G4String& materialName,const G4String& componentOrdinal,const G4String& line)
{
  // Reads the 1st field (after the starter) as the component's name
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  G4String name = ReadComponentName(materialName,componentOrdinal,stringPair.first);
  if ( name == "auto" )
    name = materialName;
  
  // Create a creator
  GateMatComponentCreator* creator = new GateMatComponentCreator(mDatabase,name);
  
  // Read the component abundance (by fraction only)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  G4String field=stringPair.first;
  creator->fraction = ReadComponentFraction(materialName,componentOrdinal,field);
  GateMessage("Materials", 5,  
	      "GateMDBFile: " << componentOrdinal 
	      << " component is the material '" << creator->name
	      << "' (fraction= " << creator->fraction << ")\n\n");
  
  return creator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Read a new element-type material component for the current compound element
GateIByFComponentCreator* GateMDBFile::ReadIsoComponent(const G4String& elementName,const G4String& componentOrdinal,const G4String& line)
{
  // Reads the 1st field (after the starter) as the component's name
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);
  G4String name = ReadComponentName(elementName,componentOrdinal,stringPair.first);
  if ( name == "auto" )
    name = elementName;

  // Create a creator
  GateIByFComponentCreator* creator = new GateIByFComponentCreator(mDatabase,name);

  // Read the component abundance (by fraction only)
  stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  G4String field=stringPair.first;
  creator->fraction = ReadComponentFraction(elementName,componentOrdinal,field);
  GateMessage("Materials", 5,
	      "GateMDBFile: " << componentOrdinal
	      << " component is the element '" << creator->name
	      << "' (fraction= " << creator->fraction << ")\n\n");

  return creator;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read one of the optional fields for a material, i.e. the state, temperature and pressure
void GateMDBFile::ReadAllMaterialOptions(const G4String& materialName,const G4String& line,GateMaterialCreator* creator)
{
  // We loop until we have read all fields of the line
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(line,theFieldSeparator);  
  GateTokenizer::CleanUpString(stringPair.first);
  while ( stringPair.first != "") {
    ReadMaterialOption(materialName,stringPair.first,creator);
    stringPair = GateTokenizer::BreakString(stringPair.second,theFieldSeparator);
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Read one of the optional fields for a material, i.e. the state, temperature and pressure
void GateMDBFile::ReadMaterialOption(const G4String& materialName,const G4String& field,GateMaterialCreator* creator)
{
  // Decode the field prefix to know what kind of field we have
  PrefixCode prefix = ReadMaterialOptionPrefix(materialName,field);

  // Call the relevant read function
  switch (prefix) {
  case prefix_state:
    creator->state = ReadMaterialState(materialName,field);
    break;
  case prefix_temp:
    creator->temp = ReadMaterialTemp(materialName,field);
    break;
  case prefix_pressure:
    creator->pressure = ReadMaterialPressure(materialName,field);
    break;
  default:
		G4String msg = "Abnormal prefix code found for an option field of material '" + materialName + "'!";
    G4Exception( "GateMDBFile::ReadMaterialOptions", "ReadMaterialOptions", FatalException, msg );
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Go through the whole file until we find a specific section
// The section name we'll look for is "[name]"
void GateMDBFile::FindSection(const G4String& name)
{
  if (LookForText("[" + name + "]")==false) {
    /*   G4Exception("\n!!! GateMDBFile["+fileName+"]::FindSection:   \n"
	 "\tI could not find any section [" + name + "] in the material database file. \n"
	 "\tYou should check the database file.\n"
	 "\tComputation aborted!!!\n");
    */ }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Go through the whole DB file from the start
// until we find a specific text (at beginning of line)
// Once the text is found, we're positioned just below the text's line
G4bool GateMDBFile::LookForText(const G4String& text)
{
  G4int len = text.length() ;
  G4String lineBuf;

  if (dbStream.fail()) {
    dbStream.close();
    dbStream.clear();  
    dbStream.open(fileName);
    if (!dbStream) {
      std::cerr << "ERROR Could not reopen " << fileName << Gateendl;
      exit(0);
    }
  }

  dbStream.seekg(0,std::ios::beg);  // Rewind!
  

  while (!dbStream.eof()) {
    if (ReadLine(lineBuf)) return false; // Error or EOF
   
    if ( strncmp(text.c_str(),lineBuf.c_str(),len)==0 )
      return true;// Line starts with the text we're looking for
  }
  return false;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Goes into a specific section of the DB file, then looks
// for a specific item.
// If the item is "Item", the routine will look through the
// sectoin until it finds a line starting with "Item:"
G4String GateMDBFile::ReadItem(const G4String& sectionName,const G4String& itemName)
{

  
  // Go in the relevant section of the DB file
  FindSection(sectionName);
  
  // Prepare the lookup variables and buffer
  G4bool flagElementFound=false;
  G4String text = itemName + ":" ;
  G4int len = text.length() ;
  G4String lineBuf;
  
   
  do {	// Read lines until we find the item or reach the OEF or the next section
    if (ReadNonEmptyLine(lineBuf)) 
      break;   // Reached EOF or read-error
    if (lineBuf.at(0)=='[') 
      break; // Reached next section
    //if ( text.compare(lineBuf,0,len) == 0 ) {
    if ( strncmp(text.c_str(),lineBuf.c_str(),len) == 0 ) {
      // We found the item we were looking for  
      flagElementFound = true;
      break;
    }
  } while (1);

  if (!flagElementFound) {  // We reached the EOF or the next section
    // GateMessage("Materials", 3, "GateMDBFile<" << fileName
    // 		<< ">::ReadItem: I could NOT find the item '"
    // 		<< itemName << "' in section ["
    // 		<< sectionName << "] of the material database. \n\n");
    return theReadItemErrorMsg;
  }

  GateMessage("Materials", 2, "GateMDBFile<" << fileName
	      << ">::ReadItem: I find the item '"
	      << itemName << "' in section ["
	      << sectionName << "] of the material database. \n\n");

  // We found the item: we return the text after the colon
  return lineBuf.substr(len);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Reads the next non-empty line from the DB file, and stores it
// in the string
// Returns 0 if everything went OK, 1 if there was any failure (including EOF) 
G4int GateMDBFile::ReadNonEmptyLine(G4String& lineBuffer)
{
  do {
    if (ReadLine(lineBuffer))  
      return 1;
    GateTokenizer::CleanUpString(lineBuffer);
  } while (lineBuffer=="");

  return 0;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Reads the next line from the DB file, and stores it
// in the string
// Returns 0 if everything went OK, 1 if there was any failure (including EOF) 
G4int GateMDBFile::ReadLine(G4String& lineBuffer)
{
  static char buffer[GATE_BUFFERSIZE];

  dbStream.getline(buffer,GATE_BUFFERSIZE);
  if (dbStream.eof())  
    return 1;
  lineBuffer = G4String(buffer);
  return 0;
}
//-----------------------------------------------------------------------------


