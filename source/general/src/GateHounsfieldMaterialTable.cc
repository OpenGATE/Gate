/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See GATE/LICENSE.txt for further details
 ----------------------*/

/*!
 \class  GateHounsfieldMaterialTable.cc
 \brief
 \author david.sarrut@creatis.insa-lyon.fr
 */

#include "GateHounsfieldMaterialTable.hh"
#include "GateDetectorConstruction.hh"
#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::GateHounsfieldMaterialTable() {
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::~GateHounsfieldMaterialTable() {
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end();) {
		it = mMaterialsVector.erase(it);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::AddMaterial(double H1, double H2, double d,
		GateHounsfieldMaterialProperties * p) {

	// Check
	if (H1 >= H2)
		GateError(
				"Error in GateHounsfieldMaterialProperties::AddMaterial " << H1 << " " << H2 << Gateendl);

	// Set values
	mMaterials mat;
	mat.mH1 = H1;
	mat.mH2 = H2;
	mat.md1 = d;

	// Material's name
	mat.mName = p->GetName() + "_" + DoubletoString(mMaterialsVector.size());

	// Create new material
	mat.mMaterial = new G4Material(mat.mName, d, p->GetNumberOfElements());

	// Material's elements
	for (GateHounsfieldMaterialProperties::iterator it = p->begin();
			it != p->end(); ++it) {
		mat.mMaterial->AddElement(it->Element, it->Fraction);
	}

	// Set material
	mMaterialsVector.push_back(mat);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterialDatabase(G4String filename) {
	std::ofstream os;
	OpenFileOutput(filename, os);
	os << "[Materials]\n";
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it) {
		os << "# Material corresponding to H=[ " << it->mH1 << ";" << it->mH2 // << "],with density=["
				//        << G4BestUnit(md1[i],"Volumic Mass")
				//        << ";" << G4BestUnit(md2[i],"Volumic Mass") << "]"
				<< " ]\n";
		WriteMaterial(*it, os);
		os << Gateendl;
	}
	os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterialtoHounsfieldLink(
		G4String filename) {
	std::ofstream os;
	OpenFileOutput(filename, os);
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it) {
		os << it->mH1 << " " << it->mH2 << " " << it->mName << Gateendl;
	}
	os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterial(G4Material * m,
		std::ofstream & os) {
	os << m->GetName() << ": d=" << G4BestUnit(m->GetDensity(), "Volumic Mass")
			<< "; n=" << m->GetNumberOfElements() << "; \n";
	for (unsigned int j = 0; j < m->GetNumberOfElements(); ++j) {
		os << "+el: name=" << m->GetElement(j)->GetName() << "; f="
				<< m->GetFractionVector()[j] << Gateendl;
	}
	//os.close();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::WriteMaterial(mMaterials m,
		std::ofstream & os) {
	os << m.mName << ": d=" << G4BestUnit(m.md1, "Volumic Mass") << "; n="
			<< m.mMaterial->GetNumberOfElements() << "; \n";
	for (unsigned int j = 0; j < m.mMaterial->GetNumberOfElements(); ++j) {
		os << "+el: name=" << m.mMaterial->GetElement(j)->GetName() << "; f="
				<< m.mMaterial->GetFractionVector()[j] << Gateendl;
	}
	//os.close();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::AddMaterial(double H1, double H2,
		G4String name) {
	mMaterials mat;
	mat.mH1 = H1;
	mat.mH2 = H2;
	mat.mName = name;
	mat.mMaterial = theMaterialDatabase.GetMaterial(name);
	mat.md1 = mat.mMaterial->GetDensity();
	mMaterialsVector.push_back(mat);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::Reset() {
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end();) {
		it = mMaterialsVector.erase(it);
	}
	mMaterialsVector.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldMaterialTable::MapLabelToMaterial(
		LabelToMaterialNameType & m) {
	m.clear();
	int i = 0;
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it, ++i) {
		// GateMessage("Core", 0,
		//               "i= " << i << " mi = "
		//             << m[i] << " mnamei = "
		//              << mName[i] << Gateendl);
		std::pair<LabelType, G4String> lMaterial(i, it->mName);
		m.insert(lMaterial);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldMaterialTable::LabelType GateHounsfieldMaterialTable::GetLabelFromH(
		double h) {
	int i = 0;
	while ((i < GetNumberOfMaterials() && h >= mMaterialsVector[i].mH1))
		++i;
	i--;
	//instead of returning a nonexisting label, return last known label
	//if ((i==GetNumberOfMaterials()-1) && h>mMaterialsVector[i].mH2) return i+1;
	return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldMaterialTable::GetHMeanFromLabel(int l) {
	double h = (mMaterialsVector[l].mH1 + mMaterialsVector[l].mH2) / 2.0;
	return h;
}
//-----------------------------------------------------------------------------
