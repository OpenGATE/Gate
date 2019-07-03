/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateMDBFieldReader_hh
#define GateMDBFieldReader_hh

#include "globals.hh"

#include "G4Material.hh"

#include "GateMDBFieldDecoder.hh"

class GateMDBFieldReader : public GateMDBFieldDecoder
{
public:
  inline GateMDBFieldReader() {}
  virtual inline ~GateMDBFieldReader() {}

public:
  enum ElementType {
    elementtype_error,
    elementtype_scratch,
    elementtype_compound
    };
  enum MaterialType {
    materialtype_error,
    materialtype_scratch,
    materialtype_compound
    };
  enum ComponentType {
    componenttype_error,
    componenttype_elem, 
    componenttype_mat,
    componenttype_iso
    };
  enum ElemComponentType {
    elemcomponent_error,
    elemcomponent_byNAtoms,
    elemcomponent_byFraction
    };
    
protected:
  PrefixCode ReadMaterialOptionPrefix(const G4String& materialName,const G4String& field);
  ElemComponentType EvaluateElemComponentType(const G4String& materialName, const G4String& field,
      	      	      	      	      	      const G4String& componentOrdinal);
  ComponentType EvaluateComponentType(const G4String& materialName, G4String field,const G4String& componentOrdinal);
  ElementType EvaluateElementType(const G4String& elementName, const G4String& field);
  MaterialType EvaluateMaterialType(const G4String& materialName, const G4String& field);

  G4String ReadElementSymbol(const G4String& elementName, const G4String& field);
  G4double ReadAtomicNumber(const G4String& elementName, const G4String& field);
  G4double ReadNucleonNumber(const G4String& elementName, const G4String& field);
  G4double ReadMolarMass(const G4String& elementName, const G4String& field);
  G4double ReadDensity(const G4String& materialName, const G4String& field);
  G4State  ReadMaterialState(const G4String& materialName, const G4String& field);
  G4double ReadMaterialTemp(const G4String& materialName, const G4String& field);
  G4double ReadMaterialPressure(const G4String& materialName, const G4String& field);
  G4int    ReadNumberOfComponents(const G4String& materialName, const G4String& field);
  G4String ReadComponentName(const G4String& componentName, const G4String& componentOrdinal, const G4String& field);
  G4int    ReadComponentNAtoms(const G4String& componentName, const G4String& componentOrdinal, const G4String& field);
  G4double ReadComponentFraction(const G4String& componentName, const G4String& componentOrdinal, const G4String& field);

  G4String     CreateOrdinalString(G4int index);

protected:
  static GateCodeMap theSymbolPrefixMap;
  static GateCodeMap theAtomicNumberPrefixMap;
  static GateCodeMap theNucleonNumberPrefixMap;
  static GateCodeMap theMolarMassPrefixMap;
  static GateCodeMap theDensityPrefixMap;
  static GateCodeMap theStatePrefixMap;
  static GateCodeMap theTempPrefixMap;
  static GateCodeMap thePressurePrefixMap;
  static GateCodeMap theNComponentsPrefixMap;
  static GateCodeMap theNamePrefixMap;
  static GateCodeMap theNAtomsPrefixMap;
  static GateCodeMap theFractionPrefixMap;

  static GateUnitMap theAtomicNumberUnitMap;
  static GateUnitMap theNucleonNumberUnitMap;
  static GateUnitMap theMolarMassUnitMap;
  static GateUnitMap theDensityUnitMap;
  static GateUnitMap theTempUnitMap;
  static GateUnitMap thePressureUnitMap;
  static GateUnitMap theFractionUnitMap;

  static GateCodeMap theFMFPrefixMap;
  static GateCodeMap theFEFPrefixMap;
  static GateCodeMap theAbundancePrefixMap;
  static GateCodeMap theMaterialOptionPrefixMap;



private:

  static GateCodePair theStateCodeTable[];

  static GateCodePair theSymbolPrefixTable[];
  static GateCodePair theAtomicNumberPrefixTable[];
  static GateCodePair theNucleonNumberPrefixTable[];
  static GateCodePair theMolarMassPrefixTable[];
  static GateCodePair theDensityPrefixTable[];
  static GateCodePair theStatePrefixTable[];
  static GateCodePair theTempPrefixTable[];
  static GateCodePair thePressurePrefixTable[];
  static GateCodePair theNComponentsPrefixTable[];
  static GateCodePair theNamePrefixTable[];
  static GateCodePair theNAtomsPrefixTable[];
  static GateCodePair theFractionPrefixTable[];
  
  static GateUnitPair theAtomicNumberUnitTable[];
  static GateUnitPair theNucleonNumberUnitTable[];
  static GateUnitPair theMolarMassUnitTable[];
  static GateUnitPair theDensityUnitTable[];
  static GateUnitPair theTempUnitTable[];
  static GateUnitPair thePressureUnitTable[];
  static GateUnitPair theFractionUnitTable[];

  static GateCodeMap* theFMFPrefixMapArray[];
  static GateCodeMap* theFEFPrefixMapArray[];
  static GateCodeMap* theAbundancePrefixMapArray[];
  static GateCodeMap* theMaterialOptionPrefixMapArray[];

};

#endif
