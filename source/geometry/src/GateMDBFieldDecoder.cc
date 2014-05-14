/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateMDBFieldDecoder.hh"
#include "GateTokenizer.hh"

char GateMDBFieldDecoder::thePrefixSeparator  = '=';
char GateMDBFieldDecoder::theUnitOptionalPrefixChar = '*';


//-------------------------------------------------------------------------------------------------
// We decode the content of an integer field, using predefined prefix and unit maps
G4double GateMDBFieldDecoder::DecodeNumericField(const G4String& elementName,G4String field,const G4String& fieldName,
						 GateCodeMap& prefixMap, GateUnitMap& unitMap)
{
  GateTokenizer::CleanUpString(field); // Remove leading/trailing spaces/tabs
  if (field == "" ) 
    DecodingException(elementName,"\tI can't find the element's " + fieldName + "\n");

  // Decode field prefix
  G4String fieldAfterPrefix;
  DecodeFieldPrefix(elementName,field, fieldName, prefixMap, fieldAfterPrefix);

  // Decode the field value
  GateTokenizer::CleanUpString(fieldAfterPrefix);
  G4String unitString;
  G4double fieldValue = DecodeFieldFloatingValue(elementName, fieldAfterPrefix, fieldName, unitString);

  // Decode the field unit
  G4double unitValue = DecodeFieldUnit(elementName,unitString, fieldName,unitMap) ; 
  
  // Return the product value * unit
  return fieldValue * unitValue;

}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the content of an integer field, using a predefined prefix-map
G4int GateMDBFieldDecoder::DecodeIntegerField(const G4String& elementName,G4String field,const G4String& fieldName,GateCodeMap& prefixMap)
{
  GateTokenizer::CleanUpString(field); // Remove leading/trailing spaces/tabs
  if (field == "" ) 
    DecodingException(elementName,"\tI can't find the element's " + fieldName + "\n");

  // Decode field prefix
  G4String fieldAfterPrefix;
  DecodeFieldPrefix(elementName,field, fieldName, prefixMap, fieldAfterPrefix);

  // Decode and return the field value
  GateTokenizer::CleanUpString(fieldAfterPrefix);
  G4int fieldValue = DecodeFieldIntegerValue(elementName, fieldAfterPrefix, fieldName);
  return fieldValue;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the content of a text field, using a predefined prefix-map
G4String GateMDBFieldDecoder::DecodeTextField(const G4String& elementName,G4String field,const G4String& fieldName,GateCodeMap& prefixMap)
{
  GateTokenizer::CleanUpString(field); // Remove leading/trailing spaces/tabs
  if (field == "" ) 
    DecodingException(elementName,"\tI can't find the element's " + fieldName + "\n");

  // Decode field prefix
  G4String fieldAfterPrefix;
  DecodeFieldPrefix(elementName,field, fieldName, prefixMap, fieldAfterPrefix);
  
  // Anything after the prefix is the field's value: we return it (after cleanup)
  GateTokenizer::CleanUpString(fieldAfterPrefix);
  return fieldAfterPrefix;

}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the first part (prefix) of a field:
// - check whether there is indeed a prefix (i.e. a codeword followed by "=")
// - check whether this codeword is listed in the prefixMap
// - store the part of the field after the prefix in the string fieldAfterPrefix
// - return the code-number for the prefix
GateMDBFieldDecoder::PrefixCode GateMDBFieldDecoder::DecodeFieldPrefix(const G4String& elementName,G4String field, const G4String& fieldName,
								       GateCodeMap& prefixMap, G4String& fieldAfterPrefix)
{
  GateTokenizer::CleanUpString(field); // Remove leading/trailing spaces/tabs
  
  // Break field at the sign "="
  GateTokenizer::BrokenString stringPair = GateTokenizer::BreakString(field,thePrefixSeparator);
  if (stringPair.separatorPos==G4String::npos)
    DecodingException(elementName,"\tCould not find any prefix for the " + fieldName + " field.\n");

  // Extract and cleanup prefix string (not including the separator "=")
  G4String prefixString = stringPair.first;
  GateTokenizer::CleanUpString(prefixString);
  
  // Look for the prefix codeword in the prefix map
  GateCodeMap::iterator codePairIt = prefixMap.find(prefixString);
  if ( codePairIt == prefixMap.end()) 
    DecodingException(elementName,
		      "\tThe prefix found for the " + fieldName + " field ('" + prefixString +"=')\n"
		      "\tis not in the list of autorised prefixes (" + prefixMap.DumpMap(true,"=",",") + ") for this field.\n");

  // Store the part of the field following the prefix
  fieldAfterPrefix = stringPair.second;
  
  // Return the prefix code
  return (PrefixCode) codePairIt->second;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the value of the current field, assumed to be a floating number
// The part of the field following this number (assuming that the conversion worked) is
// stored into the unitString
G4double GateMDBFieldDecoder::DecodeFieldFloatingValue(const G4String& elementName, G4String fieldAfterPrefix, const G4String& fieldName,G4String& unitString)
{
  GateTokenizer::CleanUpString(fieldAfterPrefix); // Remove leading/trailing spaces/tabs

  // Convert the string to a double
  char* conversionEndPtr;
  G4double fieldValue = strtod(fieldAfterPrefix.c_str(),&conversionEndPtr) ;
  if (conversionEndPtr==fieldAfterPrefix.c_str())   
    DecodingException(elementName,"\tI couldn't find any numerical value for the " + fieldName + " field\n");
    
  // Store into unitString the content of the field from the point where the conversion stopped
  unitString = conversionEndPtr;
  
  // Return the field double value
  return fieldValue;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the value of the current field, assumed to be an integer number
// We check that nothing remains following this number (assuming that the conversion worked)
G4int GateMDBFieldDecoder::DecodeFieldIntegerValue(const G4String& elementName, G4String fieldAfterPrefix, const G4String& fieldName)
{
  GateTokenizer::CleanUpString(fieldAfterPrefix); // Remove leading/trailing spaces/tabs

  // Convert the string to an integer
  char* conversionEndPtr;
  G4int fieldValue = strtol(fieldAfterPrefix.c_str(),&conversionEndPtr,10) ;
  if (conversionEndPtr==fieldAfterPrefix.c_str())   
    DecodingException(elementName,"\tI couldn't find any numerical value for the " + fieldName + " field\n");

  // Check that there was no trailing unit part
  G4String unitString = conversionEndPtr;
  GateTokenizer::CleanUpString(unitString);
  if (unitString!="")
    DecodingException(elementName,"\tI found an erroneous string ('" + unitString 
		      + "') after the numerical value for the " + fieldName + " field\n");

  // Return the field integer value
  return fieldValue;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// We decode the unit of the current field, using a predefined unit-map, and return the corresponding
// unit value
G4double GateMDBFieldDecoder::DecodeFieldUnit(const G4String& elementName, G4String unitString, const G4String& fieldName,GateUnitMap& unitMap)
{
  GateTokenizer::CleanUpString(unitString); // Remove leading/trailing spaces/tabs
  
  //Remove a leading "*" if any
  if (!unitString.empty())
    if (unitString.at(0) == theUnitOptionalPrefixChar)
      unitString = unitString.substr(1);      
  
  
  // Look for the unit codeword in the unit map
  GateUnitMap::iterator unitPairIt = unitMap.find(unitString);
  if ( unitPairIt == unitMap.end()) 
    DecodingException(elementName,
		      "\tThe unit found for the " + fieldName + " field ('" + unitString +"')\n"
		      "\tis not in the list of autorised units (" + unitMap.DumpMap(true,"",",") + ") for this field.\n");

  // Return unit value
  return unitPairIt->second;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Common G4Exception function used by a lot of read functions
void GateMDBFieldDecoder::DecodingException(const G4String& elementName,const G4String& errorMsg)
{
	G4String msg = "The definition of the element/material '" + elementName + "' is incorrect: " + errorMsg + "You should check this definition in the database file.";
  G4Exception(  "GateMDBFieldDecoder::DecodingException", "DecodingException", FatalException, msg );
}
//-------------------------------------------------------------------------------------------------




