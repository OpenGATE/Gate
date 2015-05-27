#include "GateRangeMaterialTable.hh"
#include "GateDetectorConstruction.hh"

//-----------------------------------------------------------------------------
GateRangeMaterialTable::GateRangeMaterialTable() {
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateRangeMaterialTable::~GateRangeMaterialTable() {
	for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end();) {
		it = mMaterialsVector.erase(it);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterialDatabase(G4String filename) {
	std::ofstream os;
	OpenFileOutput(filename, os);
	os << "[Materials]\n";
	for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it) {
		os << "# Material corresponding to H=[ " << it->mR1 << ";" << it->mR2 // << "],with density=["
				//        << G4BestUnit(md1[i],"Volumic Mass")
				//        << ";" << G4BestUnit(md2[i],"Volumic Mass") << "]"
				<< " ]\n";
		WriteMaterial(it->mMaterial, os);
		os << Gateendl;
	}
	os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterialtoRangeLink(G4String filename) {
	std::ofstream os;
	OpenFileOutput(filename, os);
	for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it) {
		os << it->mR1 << " " << it->mR2 << " " << it->mMaterial->GetName()
				<< Gateendl;
	}
	os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::WriteMaterial(G4Material * m, std::ofstream & os) {
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
void GateRangeMaterialTable::AddMaterial(int R1, int R2, G4String name,
		G4VisAttributes* Attributes) {
	mMaterials mat;
	mat.mR1 = R1;
	mat.mR2 = R2;
	mat.mName = name;
	mat.mMaterial = theMaterialDatabase.GetMaterial(name);
	mat.md1 = mat.mMaterial->GetDensity();
	mat.mVisAttributes = Attributes;
	mMaterialsVector.push_back(mat);
	GateMessage("Materials", 2,
			"Material added to Database:" << mat.mName << "; density = " << mat.md1/(gram/centimeter3) << " g/cm3" << Gateendl);
}
//-----------------------------------------------------------------------------
void GateRangeMaterialTable::AddMaterial(int R1, int R2, G4String name,
		G4bool visibility, G4Colour Color) {
	AddMaterial(R1, R2, name, new G4VisAttributes(visibility, Color));
}

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::Reset() {
	for (GateMaterialsVector::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end();) {
		it = mMaterialsVector.erase(it);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateRangeMaterialTable::MapLabelToMaterial(LabelToMaterialNameType & m) {
	m.clear();
	int i = 0;
	for (std::vector<mMaterials>::iterator it = mMaterialsVector.begin();
			it != mMaterialsVector.end(); ++it, ++i) {
		std::pair<LabelType, G4String> lMaterial(i, it->mName);
		m.insert(lMaterial);
		GateMessage("Core", 2,
				"i= " << i << " mi = " << m[i] << " mnamei = " << it->mName << " dens = " << it->md1/(gram/centimeter3) << " g/cm3" << Gateendl);
	}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateRangeMaterialTable::LabelType GateRangeMaterialTable::GetLabelFromR(int h) {
	int i = 0;
	while ((i < GetNumberOfMaterials() && h >= mMaterialsVector[i].mR1))
		++i;
	i--;
	if ((i == GetNumberOfMaterials() - 1) && h > mMaterialsVector[i].mR2)
		return i + 1;
	return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateRangeMaterialTable::GetRMeanFromLabel(int l) {
	double h = (mMaterialsVector[l].mR1 + mMaterialsVector[l].mR2) / 2.0;
	return h;
}
//-----------------------------------------------------------------------------
