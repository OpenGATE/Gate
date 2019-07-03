/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



/*!
  \file GateMDBFieldReader.cc
  
  \brief Class GateMDBFieldReader
*/

#include "G4SystemOfUnits.hh"

#include "GateMDBFieldReader.hh"
#include "GateTokenizer.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#define N_STATECODES 12
GateCodePair GateMDBFieldReader::theStateCodeTable[N_STATECODES] = { 
    GateCodePair("kStateUndefined",kStateUndefined) , 
    GateCodePair("kStateSolid",    kStateSolid) , 
    GateCodePair("kStateLiquid",   kStateLiquid) , 
    GateCodePair("kStateGas",      kStateGas) , 
    GateCodePair("Undefined",  	  kStateUndefined) , 
    GateCodePair("Solid",      	  kStateSolid) , 
    GateCodePair("Liquid",     	  kStateLiquid) , 
    GateCodePair("gas",        	  kStateGas), 
    GateCodePair("undefined",  	  kStateUndefined) , 
    GateCodePair("solid",      	  kStateSolid) , 
    GateCodePair("liquid",     	  kStateLiquid) , 
    GateCodePair("fas",        	  kStateGas) 
  };




#define N_SYMBOLPREFIX 3
GateCodePair GateMDBFieldReader::theSymbolPrefixTable[N_SYMBOLPREFIX] = {
    GateCodePair ("S",        prefix_symbol),
    GateCodePair ("Symbol",     prefix_symbol),
    GateCodePair ("symbol",     prefix_symbol)
  };  
GateCodeMap GateMDBFieldReader::theSymbolPrefixMap = GateCodeMap(N_SYMBOLPREFIX,theSymbolPrefixTable);

  
#define N_ATOMICNUMBERPREFIX 2
GateCodePair GateMDBFieldReader::theAtomicNumberPrefixTable[N_ATOMICNUMBERPREFIX] = {
    GateCodePair ("Z",        prefix_atomicnumber),
    GateCodePair ("Zeff",     prefix_atomicnumber)
  };  
GateCodeMap GateMDBFieldReader::theAtomicNumberPrefixMap = GateCodeMap(N_ATOMICNUMBERPREFIX,theAtomicNumberPrefixTable);


#define N_NUCLEONNUMBERPREFIX 2
GateCodePair GateMDBFieldReader::theNucleonNumberPrefixTable[N_NUCLEONNUMBERPREFIX] = {
    GateCodePair ("N",        prefix_nucleonnumber),
    GateCodePair ("Neff",     prefix_nucleonnumber)
  };
GateCodeMap GateMDBFieldReader::theNucleonNumberPrefixMap = GateCodeMap(N_NUCLEONNUMBERPREFIX,theNucleonNumberPrefixTable);


#define N_MOLARMASSPREFIX 2
GateCodePair GateMDBFieldReader::theMolarMassPrefixTable[N_MOLARMASSPREFIX] = {
    GateCodePair ("A",        prefix_molarmass),
    GateCodePair ("Aeff",     prefix_molarmass)
  };  
GateCodeMap GateMDBFieldReader::theMolarMassPrefixMap = GateCodeMap(N_MOLARMASSPREFIX,theMolarMassPrefixTable);

  
#define N_DENSITYPREFIX 4
GateCodePair GateMDBFieldReader::theDensityPrefixTable[N_DENSITYPREFIX] = {
    GateCodePair ("d",        prefix_density),
    GateCodePair ("D",        prefix_density),
    GateCodePair ("density",  prefix_density),
    GateCodePair ("Density",  prefix_density)
  };  
GateCodeMap GateMDBFieldReader::theDensityPrefixMap = GateCodeMap(N_DENSITYPREFIX,theDensityPrefixTable);

  
#define N_STATEPREFIX 2
GateCodePair GateMDBFieldReader::theStatePrefixTable[N_STATEPREFIX] = {
    GateCodePair ("state",  prefix_state),
    GateCodePair ("State",  prefix_state)
  };  
GateCodeMap  GateMDBFieldReader::theStatePrefixMap = GateCodeMap(N_STATEPREFIX,theStatePrefixTable);
  
#define N_TEMPPREFIX 6
GateCodePair GateMDBFieldReader::theTempPrefixTable[N_TEMPPREFIX] = { 
    GateCodePair("t",          	 prefix_temp) , 
    GateCodePair("T",          	 prefix_temp) ,
    GateCodePair("temp",       	 prefix_temp) , 
    GateCodePair("Temp",       	 prefix_temp) ,
    GateCodePair("temperature",   prefix_temp) , 
    GateCodePair("Temperature",   prefix_temp) 
  };
GateCodeMap GateMDBFieldReader::theTempPrefixMap = GateCodeMap(N_TEMPPREFIX,theTempPrefixTable);

#define N_PRESSUREPREFIX 6
GateCodePair GateMDBFieldReader::thePressurePrefixTable[N_PRESSUREPREFIX] = { 
    GateCodePair("p",          prefix_pressure) , 
    GateCodePair("P",          prefix_pressure) ,
    GateCodePair("pressure",   prefix_pressure) , 
    GateCodePair("Pressure",   prefix_pressure) ,
    GateCodePair("press",      prefix_pressure) , 
    GateCodePair("Press",      prefix_pressure) 
  };
GateCodeMap  GateMDBFieldReader::thePressurePrefixMap = GateCodeMap(N_PRESSUREPREFIX,thePressurePrefixTable);

#define N_NCOMPONENTSSPREFIX 6
GateCodePair GateMDBFieldReader::theNComponentsPrefixTable[N_NCOMPONENTSSPREFIX] = { 
    GateCodePair("N",          prefix_ncomponents) , 
    GateCodePair("n",          prefix_ncomponents) ,
    GateCodePair("ncomponents",prefix_ncomponents) , 
    GateCodePair("nComponents",prefix_ncomponents) ,
    GateCodePair("Ncomponents",prefix_ncomponents) , 
    GateCodePair("NComponents",prefix_ncomponents) 
  };
GateCodeMap GateMDBFieldReader::theNComponentsPrefixMap = GateCodeMap(N_NCOMPONENTSSPREFIX,theNComponentsPrefixTable);


#define N_NAMEPREFIX 2
GateCodePair GateMDBFieldReader::theNamePrefixTable[N_NAMEPREFIX] = { 
    GateCodePair("Name",     prefix_name) , 
    GateCodePair("name",     prefix_name) 
  };
GateCodeMap GateMDBFieldReader::theNamePrefixMap = GateCodeMap(N_NAMEPREFIX,theNamePrefixTable);


#define N_NATOMSPREFIX 6
GateCodePair GateMDBFieldReader::theNAtomsPrefixTable[N_NATOMSPREFIX] = { 
    GateCodePair("N",     prefix_natoms) , 
    GateCodePair("n",     prefix_natoms) ,
    GateCodePair("Natoms",prefix_natoms) , 
    GateCodePair("natoms",prefix_natoms) ,
    GateCodePair("NAtoms",prefix_natoms) , 
    GateCodePair("nAtoms",prefix_natoms) 
  };
GateCodeMap GateMDBFieldReader::theNAtomsPrefixMap = GateCodeMap(N_NATOMSPREFIX,theNAtomsPrefixTable);

#define N_FRACTIONPREFIX  4
GateCodePair GateMDBFieldReader::theFractionPrefixTable[N_FRACTIONPREFIX] = { 
    GateCodePair("F",        prefix_fraction) , 
    GateCodePair("f",        prefix_fraction) ,
    GateCodePair("Fraction", prefix_fraction) , 
    GateCodePair("fraction", prefix_fraction)
  };
GateCodeMap GateMDBFieldReader::theFractionPrefixMap = GateCodeMap(N_FRACTIONPREFIX,theFractionPrefixTable);



#define N_ATOMICNUMBERUNIT 1
GateUnitPair GateMDBFieldReader::theAtomicNumberUnitTable[N_ATOMICNUMBERUNIT] = {
    GateUnitPair("" , 1)
  };
GateUnitMap GateMDBFieldReader::theAtomicNumberUnitMap = GateUnitMap(N_ATOMICNUMBERUNIT,theAtomicNumberUnitTable);


#define N_NUCLEONNUMBERUNIT 1
GateUnitPair GateMDBFieldReader::theNucleonNumberUnitTable[N_NUCLEONNUMBERUNIT] = {
    GateUnitPair("" , 1)
  };
GateUnitMap GateMDBFieldReader::theNucleonNumberUnitMap = GateUnitMap(N_NUCLEONNUMBERUNIT,theNucleonNumberUnitTable);


#define N_MOLARMASSUNIT 1
GateUnitPair GateMDBFieldReader::theMolarMassUnitTable[N_MOLARMASSUNIT] = {
    GateUnitPair("g/mole" , gram/mole)
  };
GateUnitMap GateMDBFieldReader::theMolarMassUnitMap = GateUnitMap(N_MOLARMASSUNIT,theMolarMassUnitTable);

#define N_DENSITYUNIT 9
GateUnitPair GateMDBFieldReader::theDensityUnitTable[N_DENSITYUNIT] = {
    GateUnitPair("mg/mm3" , mg/mm3),
    GateUnitPair("g/mm3" ,  g /mm3),
    GateUnitPair("kg/mm3" , kg/mm3),
    GateUnitPair("mg/cm3" , mg/cm3),
    GateUnitPair("g/cm3" ,  g /cm3),
    GateUnitPair("kg/cm3" , kg/cm3),
    GateUnitPair("mg/m3" ,  mg/m3),
    GateUnitPair("g/m3" ,   g /m3),
    GateUnitPair("kg/m3" ,  kg/m3)
  };
GateUnitMap GateMDBFieldReader::theDensityUnitMap = GateUnitMap(N_DENSITYUNIT,theDensityUnitTable);

#define N_TEMPUNIT 2
GateUnitPair GateMDBFieldReader::theTempUnitTable[N_TEMPUNIT] = {
    GateUnitPair("kelvin" ,  kelvin),
    GateUnitPair("Kelvin" ,  kelvin)
  };
GateUnitMap  GateMDBFieldReader::theTempUnitMap = GateUnitMap(N_TEMPUNIT,theTempUnitTable);

#define N_PRESSUREUNIT 3
GateUnitPair GateMDBFieldReader::thePressureUnitTable[N_PRESSUREUNIT] = {
    GateUnitPair("atm" ,  atmosphere),
    GateUnitPair("bar" ,  bar),
    GateUnitPair("pascal", pascal)
  };
GateUnitMap  GateMDBFieldReader::thePressureUnitMap = GateUnitMap(N_PRESSUREUNIT,thePressureUnitTable);

#define N_FRACTIONUNIT 8
GateUnitPair GateMDBFieldReader::theFractionUnitTable[N_FRACTIONUNIT] = {
    GateUnitPair("" ,             1),
    GateUnitPair("%" ,            perCent),
    GateUnitPair("percent" ,      perCent),
    GateUnitPair("perCent" ,      perCent),
    GateUnitPair("perthousand" ,  perThousand),
    GateUnitPair("perThousand" ,  perThousand),
    GateUnitPair("permillion" ,   perMillion),
    GateUnitPair("perMillion" ,   perMillion)
  };
GateUnitMap GateMDBFieldReader::theFractionUnitMap = GateUnitMap(N_FRACTIONUNIT,theFractionUnitTable);


#define N_FMFPREFIXMAPS 2
GateCodeMap* GateMDBFieldReader::theFMFPrefixMapArray[N_FMFPREFIXMAPS] ={
  &theAtomicNumberPrefixMap,
  &theDensityPrefixMap
  };
GateCodeMap GateMDBFieldReader::theFMFPrefixMap = GateCodeMap(N_FMFPREFIXMAPS,theFMFPrefixMapArray);


#define N_FEFPREFIXMAPS 2
GateCodeMap* GateMDBFieldReader::theFEFPrefixMapArray[N_FEFPREFIXMAPS] ={
  &theSymbolPrefixMap,
  &theNComponentsPrefixMap
  };
GateCodeMap GateMDBFieldReader::theFEFPrefixMap = GateCodeMap(N_FMFPREFIXMAPS,theFEFPrefixMapArray);


#define N_ABUNDANCEPREFIXMAPS 2
GateCodeMap* GateMDBFieldReader::theAbundancePrefixMapArray[N_ABUNDANCEPREFIXMAPS] ={
  &theNAtomsPrefixMap,
  &theFractionPrefixMap
  };
GateCodeMap GateMDBFieldReader::theAbundancePrefixMap = GateCodeMap(N_ABUNDANCEPREFIXMAPS,theAbundancePrefixMapArray);

#define N_MATERIALOPTIONPREFIXMAPS 3
GateCodeMap* GateMDBFieldReader::theMaterialOptionPrefixMapArray[N_MATERIALOPTIONPREFIXMAPS] ={
  &theStatePrefixMap,
  &theTempPrefixMap,
  &thePressurePrefixMap
  };
GateCodeMap GateMDBFieldReader::theMaterialOptionPrefixMap = GateCodeMap(N_MATERIALOPTIONPREFIXMAPS,theMaterialOptionPrefixMapArray);




// Read the prefix of an optional field (state, temperature and pressure) for a material
GateMDBFieldReader::PrefixCode GateMDBFieldReader::ReadMaterialOptionPrefix(const G4String& materialName,const G4String& field)
{
  G4String fieldAfterPrefix;
  return DecodeFieldPrefix(materialName,field, "option", theMaterialOptionPrefixMap, fieldAfterPrefix);
}


// Read the prefix of the last field (abundance) of an element-type compound material'component line to check
// what kind of element-type component it will be (i.e. by fraction or by number)
// The appropriate element-type component type-code is returned.
GateMDBFieldReader::ElemComponentType GateMDBFieldReader::EvaluateElemComponentType(const G4String& materialName, const G4String& field,
      	      	      	      	      	      	      	      	      	            const G4String& componentOrdinal)
{
  G4String fieldAfterPrefix;
  PrefixCode fieldPrefix = DecodeFieldPrefix(materialName,field, "abundance", theAbundancePrefixMap, fieldAfterPrefix);

  switch (fieldPrefix) {
    case prefix_natoms:
      return elemcomponent_byNAtoms;
    case prefix_fraction:
      return elemcomponent_byFraction;
    default:
			G4String msg = "Abnormal prefix code found for the abundance field of ";
			msg += componentOrdinal;
			msg += " component of material '";
			msg += materialName;
			msg += "'!";
      G4Exception( "GateMDBFile::EvaluateElemComponentType", "EvaluateElemComponentType", FatalException, msg );
  }
  return elemcomponent_error;
}




// Read the very first field of a material's component line to check what kind of component it will be
// +el/+elem --> element-type component
// +mat --> material-type component
// +iso --> isotopes-type component
// The appropriate component type-code is returned.
GateMDBFieldReader::ComponentType GateMDBFieldReader::EvaluateComponentType(const G4String& materialName, G4String field,
      	      	      	      	      	      	      	      	      	    const G4String& componentOrdinal)
{
  GateTokenizer::CleanUpString(field);
  if ( (field=="+el") || (field=="+elem") )
    return componenttype_elem;
  else if (field=="+mat") 
    return componenttype_mat;
  else if (field=="+iso")
    return componenttype_iso;
  else {
	G4String msg = "Incorrect definition line for the " +  componentOrdinal +  " component of the compound material '" + materialName + "'.";
	msg += "This line should start with '+el:' or '+mat:' or '+iso:'. You should check the list of components in the database file for this material.";
	G4Exception( "GateMDBFieldReader::EvaluateComponentType", "EvaluateComponentType", FatalException, msg );
  }
  return componenttype_error;
}

// Read the prefix of the first field of a material line to check what kind of material it will be
// If this field contains an atomic number (Z), it will be a "scratch material"
// If this fields contains a density, it will be a compound material
// The appropriate material type-code is returned.
GateMDBFieldReader::ElementType GateMDBFieldReader::EvaluateElementType(const G4String& elementName, const G4String& field)
{
  G4String fieldAfterPrefix;
  PrefixCode prefix = DecodeFieldPrefix(elementName,field, "first material's", theFEFPrefixMap, fieldAfterPrefix);

  switch (prefix) {
    case prefix_symbol:
      return elementtype_scratch;
    case prefix_ncomponents:
      return elementtype_compound;
    default:
			G4String msg = "Abnormal prefix code found for the first field of element '";
			msg += elementName;
			msg += "'";
      G4Exception( "GateMDBFieldReader::EvaluateElementType", "EvaluateElementType", FatalException, msg );
  }
  return elementtype_error;
}



// Read the prefix of the first field of a material line to check what kind of material it will be
// If this field contains an atomic number (Z), it will be a "scratch material"
// If this fields contains a density, it will be a compound material
// The appropriate material type-code is returned.
GateMDBFieldReader::MaterialType GateMDBFieldReader::EvaluateMaterialType(const G4String& materialName, const G4String& field)
{
  G4String fieldAfterPrefix;
  PrefixCode prefix = DecodeFieldPrefix(materialName,field, "first material's", theFMFPrefixMap, fieldAfterPrefix);

  switch (prefix) {
    case prefix_atomicnumber:
      return materialtype_scratch;
    case prefix_density:
      return materialtype_compound;
    default:
			G4String msg = "Abnormal prefix code found for the first field of material '";
			msg += materialName;
			msg += "'";
      G4Exception( "GateMDBFieldReader::EvaluateMaterialType", "EvaluateMaterialType", FatalException, msg );
  }
  return materialtype_error;
}




// Read the symbol of an element
G4String GateMDBFieldReader::ReadElementSymbol(const G4String&  name, const G4String& field)
{
  return DecodeTextField(name, field, "symbol", theSymbolPrefixMap);
}



// Read the atomic number of an element
G4double GateMDBFieldReader::ReadAtomicNumber(const G4String& elementName, const G4String& field)
{
  return DecodeNumericField(elementName, field, "atomic number", theAtomicNumberPrefixMap, theAtomicNumberUnitMap);
}


// Read the nucleon number of an element
G4double GateMDBFieldReader::ReadNucleonNumber(const G4String& elementName, const G4String& field)
{
  return DecodeNumericField(elementName, field, "nucleon number", theNucleonNumberPrefixMap, theNucleonNumberUnitMap);
}





// Read the density of an element
G4double GateMDBFieldReader::ReadMolarMass(const G4String& elementName, const G4String& field)
{
  return DecodeNumericField(elementName, field, "molar mass", theMolarMassPrefixMap, theMolarMassUnitMap);
}





// Read the density of a material
G4double GateMDBFieldReader::ReadDensity(const G4String& materialName, const G4String& field)
{
  return DecodeNumericField(materialName, field, "density", theDensityPrefixMap, theDensityUnitMap);
}





// Read the state of a material
G4State GateMDBFieldReader::ReadMaterialState(const G4String& materialName, const G4String& field)
{
  G4String stateString = DecodeTextField(materialName, field, "state", theStatePrefixMap);

  for (G4int i=0; i<N_STATECODES ;i++)
    if ( theStateCodeTable[i].first == stateString ) {
      return (G4State) theStateCodeTable[i].second;
    }
  DecodingException(materialName,
		    "\tThe value found for the state field ('" + stateString +"') was not recognized.\n");
  return kStateUndefined;
}







// Read the temperature
G4double GateMDBFieldReader::ReadMaterialTemp(const G4String& materialName, const G4String& field)
{
  return DecodeNumericField(materialName, field, "temp", theTempPrefixMap, theTempUnitMap);
}





// Read the pressure
G4double GateMDBFieldReader::ReadMaterialPressure(const G4String& materialName, const G4String& field)
{
  return DecodeNumericField(materialName, field, "pressure", thePressurePrefixMap, thePressureUnitMap);
}






// Read the number of components in a compound material
G4int GateMDBFieldReader::ReadNumberOfComponents(const G4String& materialName, const G4String& field)
{
  return DecodeIntegerField(materialName, field, "component number", theNComponentsPrefixMap);
}





// Read the name of a material's component
G4String GateMDBFieldReader::ReadComponentName(const G4String& componentName, const G4String& componentOrdinal, const G4String& field)
{
  return DecodeTextField(componentName, field, componentOrdinal + " component's name", theNamePrefixMap);
}





// Read the abundance of a material's component as a number of atoms
G4int GateMDBFieldReader::ReadComponentNAtoms(const G4String& componentName, const G4String& componentOrdinal, const G4String& field)
{
  return DecodeIntegerField(componentName, field, componentOrdinal + "component's atom number", theNAtomsPrefixMap);
}





// Read the abundance of a material's component as a mass fraction
G4double GateMDBFieldReader::ReadComponentFraction(const G4String& componentName, const G4String& componentOrdinal, const G4String& field)
{
  return DecodeNumericField(componentName, field, componentOrdinal + "component's fraction", theFractionPrefixMap, theFractionUnitMap);
}





// Simple function to create an ordinal string ("1st", "2nd", "3rd, "4th", ...)
// from an integer number
G4String GateMDBFieldReader::CreateOrdinalString(G4int anIndex)
{
  static char buffer[256];
  
  switch (anIndex) {
  case 1:
    return "1st";
  case 2:
    return "2nd";
  case 3:
    return "3rd";
  default:
    sprintf(buffer,"%ith",anIndex);
    return buffer;
  }
}


