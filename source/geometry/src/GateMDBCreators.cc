/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \file GateMDBCreators.cc
  
  \brief Classes GateElementCreator, GateComponentCreator, GateElemComponentCreator, 
  \brief GateEByFComponentCreator,GateEByNComponentCreator, GateMatComponentCreator, GateMaterialCreator
  \brief GateScratchMaterialCreator, GateCompoundMaterialCreator
*/

#include "GateMDBCreators.hh"

//
#include "GateMaterialDatabase.hh"
#include "GateMessageManager.hh"

//-----------------------------------------------------------------------------
G4Isotope* GateIsotopeCreator::Construct()
{
  G4Isotope* isotope = new G4Isotope(name,atomicNumber,nucleonNumber,molarMass);
  if (!isotope)
	{
		G4String msg = "Failed to create a new isotope for '" + name;
    G4Exception( "GateIsotopeCreator::Construct", "Construct", FatalException, msg );
	}



  return isotope;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
G4Element* GateScratchElementCreator::Construct()
{
  G4Element* element = new G4Element(name,symbol,atomicNumber,molarMass);
  if (!element)
	{
		G4String msg = "Failed to create a new element for '" + name;
    G4Exception( "GateElementCreator::Construct", "Construct", FatalException, msg );
	}



  return element;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Element* GateCompoundElementCreator::Construct()
{
  G4Element* element = new G4Element(name,symbol,nComponents);

  if (!element)
	{
		G4String msg = "Failed to create a new element for '" + name + "'!";
    G4Exception( "GateCompoundElementCreator::Construct", "Construct", FatalException, msg );
	}
  for (G4int i=0; i<nComponents ; i++)
    components[i]->AddToElement(element);

  double f=0.0;
  for(unsigned int j=0; j<element->GetNumberOfElements(); j++) {
	double frac = element->GetRelativeAbundanceVector()[j];
	f+=frac;
	if (frac<0.0) {
	  GateError("Fraction lower than 0.0 ! "
				<< " for element " << element->GetName()
				<< " " << frac);
	}
	if (frac>1.0) {
	  GateError("Fraction greater than 1.0 ! "
				<< " for element " << element->GetName()
				<< " " << frac);
	}
  }
  if (f > 1.001 || f < 0.999) {
	GateError("Sum of fraction is not 1.0 for element "
			  << element->GetName()
			  << " " << f);
	for(unsigned int j=0; j<element->GetNumberOfElements(); j++) {
	  GateError("Element " << j << " "
				<< element->GetElement(j)->GetName()
				<< " = " << element->GetRelativeAbundanceVector()[j]);
	}
	exit(0);
  }
  return element;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIByFComponentCreator::AddToElement(G4Element* element)
{
  G4Isotope* componentIsotope = mDatabase->GetIsotope(name);
  if (!componentIsotope)
	{
		G4String msg = "Failed to retrieve component isotope '" + name + "'!";
    G4Exception( "GateIByFComponentCreator::AddToElement", "AddToElement", FatalException, msg );
	}
  element->AddIsotope(componentIsotope,fraction);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEByFComponentCreator::AddToMaterial(G4Material* material)
{
  G4Element* componentElement = mDatabase->GetElement(name);
  if (!componentElement)
	{
		G4String msg = "Failed to retrieve component element '" + name + "'!";
    G4Exception( "GateEByFComponentCreator::AddToMaterial", "AddToMaterial", FatalException, msg );
	}
  material->AddElement(componentElement,fraction);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEByNComponentCreator::AddToMaterial(G4Material* material)
{
  G4Element* componentElement = mDatabase->GetElement(name);
  if (!componentElement)
	{
		G4String msg = "Failed to retrieve component element '" + name + "'!";
    G4Exception( "GateEByNComponentCreator::AddToMaterial", "AddToMaterial", FatalException, msg );
  }
	material->AddElement(componentElement,nAtoms);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMatComponentCreator::AddToMaterial(G4Material* material)
{
  G4Material* componentMaterial = mDatabase->GetMaterial(name);
  if (!componentMaterial)
	{
		G4String msg = "Failed to retrieve component material '" + name + "'!";
    G4Exception( "GateMatComponentCreator::AddToMaterial", "AddToMaterial", FatalException, msg );
	}
  material->AddMaterial(componentMaterial,fraction);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Material* GateScratchMaterialCreator::Construct()
{
  G4Material* material = new G4Material(name,atomicNumber,molarMass,density,state,temp,pressure);
  if (!material)
	{
		G4String msg = "Failed to create a new material for '" + name + "'!";
    G4Exception( "GateScratchMaterialCreator::Construct", "Construct", FatalException, msg );
	}
  return material;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Material* GateCompoundMaterialCreator::Construct()
{
  G4Material* material = new G4Material(name,density,nComponents,state,temp,pressure);
  
  if (!material)
	{
		G4String msg = "Failed to create a new material for '" + name + "'!";
    G4Exception( "GateCompoundMaterialCreator::Construct", "Construct", FatalException, msg );
	}
  for (G4int i=0; i<nComponents ; i++)
    components[i]->AddToMaterial(material);

  double f=0.0;
  for(unsigned int j=0; j<material->GetNumberOfElements(); j++) {
	double frac = material->GetFractionVector()[j];
	f+=frac;
	if (frac<0.0) {
	  GateError("Fraction lower than 0.0 ! "
				<< " for mat " << material->GetName()
				<< " " << frac);
	}
	if (frac>1.0) {
	  GateError("Fraction greater than 1.0 ! "
				<< " for mat " << material->GetName()
				<< " " << frac);
	}
  }
  if (f > 1.001 || f < 0.999) {
	GateError("Sum of fraction is not 1.0 for mat " 
			  << material->GetName()
			  << " " << f);
	for(unsigned int j=0; j<material->GetNumberOfElements(); j++) {
	  GateError("Element " << j << " " 
				<< material->GetElement(j)->GetName()
				<< " = " << material->GetFractionVector()[j]);
	}
	exit(0);
  }
  return material;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateCompoundElementCreator::~GateCompoundElementCreator()
{
  for (std::vector<GateComponentCreator*>::iterator iter=components.begin() ; iter!= components.end() ; iter++)
    delete *iter;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateCompoundMaterialCreator::~GateCompoundMaterialCreator()
{
  for (std::vector<GateComponentCreator*>::iterator iter=components.begin() ; iter!= components.end() ; iter++)
    delete *iter;
}
//-----------------------------------------------------------------------------
