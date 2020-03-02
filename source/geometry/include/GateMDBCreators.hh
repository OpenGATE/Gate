/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/



/*!
  \file GateMDBCreators.hh
  
  \brief Classes GateElementCreator, GateComponentCreator, GateElemComponentCreator, 
  \brief GateEByFComponentCreator,GateEByNComponentCreator, GateMatComponentCreator, GateMaterialCreator
  \brief GateScratchMaterialCreator, GateCompoundMaterialCreator
*/
#ifndef GateMDBCreators_hh
#define GateMDBCreators_hh

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "GateConfiguration.h"
#include <vector>

#include "G4Material.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

class GateMaterialDatabase;

//-------------------------------------------------------------------------------------------------
class GateIsotopeCreator {
public:
  inline GateIsotopeCreator(const G4String& itsName="")
    : name(itsName),molarMass(0),atomicNumber(0),nucleonNumber(0) {}
  virtual inline ~GateIsotopeCreator() {}
  G4Isotope* Construct();


  G4String  	      	      	name;
  G4double  	      	      	molarMass;
  G4double  	      	      	atomicNumber;
  G4double  	      	      	nucleonNumber;
} ;
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateElementCreator {
public:
  inline GateElementCreator(const G4String& itsName="") 
    : name(itsName),symbol("") {}
  virtual inline ~GateElementCreator() {}
  virtual G4Element* Construct() =0;
      
    
  G4String  	      	      	name;
  G4String  	      	      	symbol;
} ;   
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateComponentCreator {
public:
  inline GateComponentCreator(  GateMaterialDatabase* db, const G4String& itsName="") : mDatabase(db), name(itsName) {}
  virtual inline ~GateComponentCreator() {}
  virtual void AddToMaterial(G4Material* material) = 0;
  virtual void AddToElement(G4Element* element) = 0;

  /// Stores the database which created this (through GateMDBFile) in order to suppress the use of static instance of the database
  GateMaterialDatabase* mDatabase;
  G4String  	  name;
} ;   
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateScratchElementCreator : public GateElementCreator {
public:
  inline GateScratchElementCreator(const G4String& itsName)
    : GateElementCreator(itsName), molarMass(0.),atomicNumber(0.) {}
  virtual ~GateScratchElementCreator() {}
  virtual G4Element* Construct();

  G4double  	      	      	molarMass;
  G4double  	      	      	atomicNumber;
} ;
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateCompoundElementCreator : public GateElementCreator {
public:
  inline GateCompoundElementCreator(const G4String& itsName)
    : GateElementCreator(itsName), nComponents(0),components() {}
  virtual ~GateCompoundElementCreator();
  virtual G4Element* Construct();

  G4int     	      	      	nComponents;
  std::vector<GateComponentCreator*>     components;
} ;
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateIsotopeComponentCreator : public GateComponentCreator {
public:
  inline GateIsotopeComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateComponentCreator(db,itsName) {}
  virtual inline ~GateIsotopeComponentCreator() {}
  virtual void AddToMaterial(G4Material* material) {UNUSED(material);}
  virtual void AddToElement(G4Element* element)=0;
} ;
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateElemComponentCreator : public GateComponentCreator {
public:
  inline GateElemComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateComponentCreator(db,itsName) {}
  virtual inline ~GateElemComponentCreator() {}
  virtual void AddToMaterial(G4Material* material)=0;
  virtual void AddToElement(G4Element* element) {UNUSED(element);}
} ;
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateIByFComponentCreator : public GateIsotopeComponentCreator {
public:
  inline GateIByFComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateIsotopeComponentCreator(db,itsName),fraction(0.) {}
  virtual inline ~GateIByFComponentCreator() {}
  virtual void AddToElement(G4Element* element);

public:
  G4double  	  fraction;
} ;   
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateEByFComponentCreator : public GateElemComponentCreator {
public:
  inline GateEByFComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateElemComponentCreator(db,itsName),fraction(0.) {}
  virtual inline ~GateEByFComponentCreator() {}
  virtual void AddToMaterial(G4Material* material);

public:
  G4double  	  fraction;
} ;   
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
class GateEByNComponentCreator : public GateElemComponentCreator {
public:
  inline GateEByNComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateElemComponentCreator(db,itsName),nAtoms(0) {}
  virtual inline ~GateEByNComponentCreator() {}
  virtual void AddToMaterial(G4Material* material);

public:
  G4int     	  nAtoms;
} ;   
//-------------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------------
class GateMatComponentCreator : public GateComponentCreator {
public:
  inline GateMatComponentCreator( GateMaterialDatabase* db, const G4String& itsName="") : GateComponentCreator(db,itsName),fraction(0.) {}
  virtual inline ~GateMatComponentCreator() {}
  virtual void AddToMaterial(G4Material* material);
  virtual void AddToElement(G4Element* element) {UNUSED(element);}

public:
  G4double  	  fraction;
} ;   
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
class GateMaterialCreator {
public:
  inline GateMaterialCreator(const G4String& itsName) 
    : name(itsName), density(0.), state(kStateUndefined), temp(STP_Temperature), pressure(STP_Pressure) {}
  virtual inline ~GateMaterialCreator() {}
  virtual G4Material* Construct()=0;
      
  G4String  	      	      	name;
  G4double  	      	      	density;
  G4State   	      	      	state;
  G4double  	      	      	temp;
  G4double  	      	      	pressure;
} ;   
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
class GateScratchMaterialCreator : public GateMaterialCreator {
public:
  inline GateScratchMaterialCreator(const G4String& itsName) 
    : GateMaterialCreator(itsName), molarMass(0.), atomicNumber(0.) {}
  virtual inline ~GateScratchMaterialCreator() {}
  virtual G4Material* Construct();
      
  G4double  	      	      	molarMass;
  G4double  	      	      	atomicNumber;
} ;   
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
class GateCompoundMaterialCreator : public GateMaterialCreator {
public:
  inline GateCompoundMaterialCreator(const G4String& itsName) 
    : GateMaterialCreator(itsName), nComponents(0),components() {}
  virtual ~GateCompoundMaterialCreator();
  virtual G4Material* Construct();
      
  G4int     	      	      	nComponents;
  std::vector<GateComponentCreator*>     components;
} ;   
//-------------------------------------------------------------------------------------------------

#endif
