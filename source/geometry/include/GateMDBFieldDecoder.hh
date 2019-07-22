/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateMDBFieldDecoder_hh
#define GateMDBFieldDecoder_hh

#include "globals.hh"

#include "GateMaps.hh"

class GateMDBFieldDecoder
{
public:
  inline GateMDBFieldDecoder() {}
  virtual inline ~GateMDBFieldDecoder() {}

public:
  enum PrefixCode {
    prefix_symbol,
    prefix_atomicnumber,
    prefix_nucleonnumber,
    prefix_molarmass,
    prefix_density,
    prefix_state,
    prefix_temp,
    prefix_pressure,
    prefix_ncomponents,
    prefix_name,
    prefix_fraction,
    prefix_natoms,
    prefix_error
    };
    
public:
  static char thePrefixSeparator;
  static char theUnitOptionalPrefixChar;

protected:
  G4double DecodeNumericField(const G4String& elementName,G4String field,const G4String& fieldName,GateCodeMap& prefixMap, GateUnitMap& unitMap);
  G4int    DecodeIntegerField(const G4String& elementName,G4String field,const G4String& fieldName,GateCodeMap& prefixMap);
  G4String DecodeTextField(const G4String& elementName,G4String field,const G4String& fieldName,GateCodeMap& prefixMap);

  PrefixCode  DecodeFieldPrefix(const G4String& elementName,G4String field, const G4String& fieldName,GateCodeMap& prefixMap, G4String& fieldAfterPrefix);
  G4double    DecodeFieldFloatingValue(const G4String& elementName, G4String fieldAfterPrefix, const G4String& fieldName,G4String& unitString);
  G4int       DecodeFieldIntegerValue(const G4String& elementName, G4String fieldAfterPrefix, const G4String& fieldName);
  G4double    DecodeFieldUnit(const G4String& elementName, G4String unitString, const G4String& fieldName,GateUnitMap& unitMap);

  void         DecodingException(const G4String& elementName,const G4String& errorMsg);

};

#endif
