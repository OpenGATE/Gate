/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See GATE/LICENSE.txt for further details
 ----------------------*/

/*!
 \class  GateHounsfieldMaterialProperties.cc
 \brief
 \author david.sarrut@creatis.insa-lyon.fr
 */

#include "GateHounsfieldMaterialProperties.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialDatabase.hh"
#include "GateDetectorConstruction.hh"

//-----------------------------------------------------------------------------
GateHounsfieldMaterialProperties::GateHounsfieldMaterialProperties() :
		mH(0) {
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialProperties::~GateHounsfieldMaterialProperties() {
	for (mElementVector::iterator it = mElementsList.begin();
			it != mElementsList.end();)
		it = mElementsList.erase(it);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateHounsfieldMaterialProperties::GetNumberOfElements() {
	return mElementsList.size();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4String GateHounsfieldMaterialProperties::GetName() {
	return mName;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldMaterialProperties::GetH() {
	return mH;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialProperties::Read(std::ifstream & is,
		std::vector<G4String> & el) {
	skipComment(is);
	if (!is)
		return;
	double h;
	is >> h;
	// Read/Create all elements fractions
	for (std::vector<G4String>::iterator it = el.begin(); it != el.end();
			++it) {
		ReadAndStoreElementFraction(is, *it);
		/*
		 //  H    C    N    O   Na  Mg   P   S   Cl  Ar  K   Ca
		 ReadAndStoreElementFraction(is, "Hydrogen");
		 ReadAndStoreElementFraction(is, "Carbon");
		 ReadAndStoreElementFraction(is, "Nitrogen");
		 ReadAndStoreElementFraction(is, "Oxygen");
		 ReadAndStoreElementFraction(is, "Sodium"); // Na
		 ReadAndStoreElementFraction(is, "Magnesium");
		 ReadAndStoreElementFraction(is, "Phosphor");
		 ReadAndStoreElementFraction(is, "Sulfur");
		 ReadAndStoreElementFraction(is, "Chlorine");
		 ReadAndStoreElementFraction(is, "Argon");
		 ReadAndStoreElementFraction(is, "Potassium");
		 ReadAndStoreElementFraction(is, "Calcium");
		 */
	}
	// Read name
	G4String n;
	is >> n;
	// Set properties
	if (is) {
		mH = h;
		mName = n;
		// Normalise fraction
		double sum = 0.0;
		//for (auto Element : mElementsList) sum += Element.Fraction; with c+11 compatibility we can do this
		for (mElementVector::iterator it = mElementsList.begin();
				it != mElementsList.end(); ++it)
			sum += it->Fraction;
		for (mElementVector::iterator it = mElementsList.begin();
				it != mElementsList.end(); ++it)
			it->Fraction /= sum;
		GateDebugMessage("Geometry", 6, "Read " << h << " " << mName << " " << sum << Gateendl);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialProperties::ReadAndStoreElementFraction(
		std::ifstream & is, G4String name) {
	skipComment(is);
	double f;
	is >> f;
	if (f > 0) {
		if (is) {
			mElementCompound EC;
			EC.Element = theMaterialDatabase.GetElement(name);
			EC.Fraction = f;
			mElementsList.push_back(EC);
		} else {
			GateError(
					"Error in reading element fraction of <" << name << ">\n");
		}
	}
}
//-----------------------------------------------------------------------------

