/*----------------------
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "G4EmParameters.hh"
#include "GateBioDoseActor.hh"
#include "GateImage.hh"
#include "GateImageWithStatistic.hh"
#include <CLHEP/Units/SystemOfUnits.h>

#define GATE_BUFFERSIZE

//-----------------------------------------------------------------------------
GateBioDoseActor::GateBioDoseActor(G4String name, G4int depth):
	GateVImageActor(name, depth),
	_currentEvent(0),
	_messenger(this),
	_alphaRef(-1),
	_betaRef(-1),
	_enableEdep(false),
	_enableDose(false),
	_enableBioDose(true),
	_enableAlphaMix(false),
	_enableBetaMix(false),
	_enableRBE(false)
{
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor() -- begin\n");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::Construct() {
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor -- Construct - begin\n");
	GateVImageActor::Construct();

	// Find G4_WATER
	G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
	// Find OtherMaterial
	//G4NistManager::Instance()->FindOrBuildMaterial(mOtherMaterial);
	G4EmParameters::Instance()->SetBuildCSDARange(true);

	// Enable callbacks BioDose
	EnableBeginOfRunAction(true);
	EnableEndOfRunAction(true);
	EnableBeginOfEventAction(true);
	EnablePreUserTrackingAction(false);
	EnablePostUserTrackingAction(false);
	EnableUserSteppingAction(true);

	// Outputs
	{
		G4String basename = removeExtension(mSaveFilename);
		G4String ext = getExtension(mSaveFilename);

		auto setupImage = [&](GateImageWithStatistic& image, std::string const& suffix) {
			G4String filename = basename + "_" + suffix + "." + ext;

			SetOriginTransformAndFlagToImage(image);
			image.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
			image.Allocate();
			image.SetFilename(filename);
		};

		if(_enableEdep)     setupImage(_edepImage, "edep");
		if(_enableDose)     setupImage(_doseImage, "dose");
		if(_enableBioDose)  setupImage(_bioDoseImage, "biodose");
		if(_enableAlphaMix) setupImage(_alphaMixImage, "alphamix");
		if(_enableBetaMix)  setupImage(_betaMixImage, "betamix");
		if(_enableRBE)      setupImage(_RBEImage, "rbe");
	}

	ResetData();

	///////////////////////////////////////////////////////////////////////////////////////////
	//Just matrix information
	G4cout << "Memory space to store physical dose into " << mResolution.x() * mResolution.y() * mResolution.z() << " voxels has been allocated " << G4endl;

	// SOBP
	if(_SOBPWeight == 0) { _SOBPWeight = 1; }

	//Building the cell line information
	_dataBase = "data/" + _cellLine + "_" + _bioPhysicalModel + ".db";
	BuildDatabase();

	if(_alphaRef < 0 || _betaRef < 0)
		GateError("BioDoseActor " << GetName() << ": setAlphaRef and setBetaRef must be done");
}
//-----------------------------------------------------------------------------
//----------------------------------------------------------------------------- // ok bio dose
void GateBioDoseActor::BuildDatabase() {
	std::ifstream f(_dataBase);
	if(!f) GateError("BioDoseActor " << GetName() << ": unable to open file '" << _dataBase << "'");

	int nZ = 0;
	double prevKineticEnergy = 1, prevAlpha = 1, prevBeta =1;

	for(std::string line; std::getline(f, line); ) {
		std::istringstream iss(line);
		std::string firstCol;

		iss >> firstCol;

		if(firstCol == "Fragment") {
			if(nZ != 0) // prevKineticEnergy is the maximum kinetic energy for current nZ
				_energyMaxForZ[nZ] = prevKineticEnergy;

			iss >> nZ;
			prevKineticEnergy = 1;
			prevAlpha = 1;
			prevBeta = 1;
		} else if(nZ != 0) {
			double kineticEnergy, alpha, beta;
			std::istringstream{firstCol} >> kineticEnergy;
			iss >> alpha >> beta;

			auto alphaCoeff = Interpol(prevKineticEnergy, kineticEnergy, prevAlpha, alpha);
			auto sqrtBetaCoeff = Interpol(prevKineticEnergy, kineticEnergy, std::sqrt(prevBeta), std::sqrt(beta));

			// Saving the in the input databse
			Fragment fragment{nZ, kineticEnergy};
			_alphaBetaInterpolTable[fragment] = {alphaCoeff, sqrtBetaCoeff};

			prevKineticEnergy = kineticEnergy;
			prevAlpha = alpha;
			prevBeta = beta;
		} else {
			GateError("BioDoseActor " << GetName() << ": bad database format in '" << _dataBase << "'");
		}
	}

	if(nZ != 0) // last line read; prevKineticEnergy is the maximum kinetic energy for current nZ
		_energyMaxForZ[nZ] = prevKineticEnergy;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
GateBioDoseActor::Coefficients GateBioDoseActor::Interpol(double x1, double x2, double y1, double y2) {
	//Function for a 1D linear interpolation. It returns a pair of a and b coefficients
	double a = (y2 - y1) / (x2 - x1);
	double b = y1 - x1 * a;
	return {a, b};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::SaveData() {
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor::SaveData() known ion events / total events: " << _eventWithKnownIonCount << " / " << _eventCount << "\n");

	// Calculate Alpha Beta mix
	for(auto const& [index, deposited]: _depositedMap) {
		// Alpha Beta mix (final)
		double alphamix = 0;
		double betamix = 0;

		if(deposited.energy != 0) {
			alphamix = (deposited.alpha / deposited.energy);
			betamix = (deposited.sqrtBeta / deposited.energy) * (deposited.sqrtBeta / deposited.energy);
		}

		// Calculate biological dose and RBE
		double biodose  = 0;
		double rbe      = 0;

		if(deposited.dose > 0 && alphamix != 0 && betamix != 0) {
			auto const sqAlphaRef = _alphaRef * _alphaRef;
			auto const sqDose     = deposited.dose * deposited.dose;
			biodose = (-_alphaRef + std::sqrt(sqAlphaRef + 4 * _betaRef * (alphamix * deposited.dose + betamix * sqDose))) / (2 * _betaRef);

			rbe = biodose / deposited.dose;
		}

		// Write data
		if(_enableEdep)     _edepImage.SetValue(index, deposited.energy);
		if(_enableDose)     _doseImage.SetValue(index, deposited.dose);
		if(_enableBioDose)  _bioDoseImage.SetValue(index, biodose);
		if(_enableAlphaMix) _alphaMixImage.SetValue(index, alphamix);
		if(_enableBetaMix)  _betaMixImage.SetValue(index, betamix);
		if(_enableRBE)      _RBEImage.SetValue(index, rbe);
	}
	//-------------------------------------------------------------
	GateVActor::SaveData();

	if(_enableEdep)     _edepImage.SaveData(_currentEvent);
	if(_enableDose)     _doseImage.SaveData(_currentEvent);
	if(_enableBioDose)  _bioDoseImage.SaveData(_currentEvent);
	if(_enableAlphaMix) _alphaMixImage.SaveData(_currentEvent);
	if(_enableBetaMix)  _betaMixImage.SaveData(_currentEvent);
	if(_enableRBE)      _RBEImage.SaveData(_currentEvent);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::BeginOfRunAction(const G4Run* r) {
	GateVActor::BeginOfRunAction(r);
	GateDebugMessage("Actor", 3, "GateBioDoseActor -- Begin of Run\n");
}
//-----------------------------------------------------------------------------
//----------------------------------------------------------------------------- // PAS DANS DOSE ACTOR
void GateBioDoseActor::EndOfRunAction(const G4Run* r) {
	GateVActor::EndOfRunAction(r);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Callback at each event
void GateBioDoseActor::BeginOfEventAction(const G4Event* e) {
	GateVActor::BeginOfEventAction(e);
	++_currentEvent;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
	double const weight     = step->GetTrack()->GetWeight();
	double const energyDep  = step->GetTotalEnergyDeposit() * weight;

	if(energyDep == 0)  return;
	if(index < 0)       return;

	DepositedMap::iterator it = _depositedMap.find(index);
	if(it == std::end(_depositedMap)) {
		_depositedMap[index] = {0, 0, 0, 0};
		it = _depositedMap.find(index);
	}

	auto& deposited = (*it).second;

	// Accumulate energy inconditionnaly
	deposited.energy += energyDep;

	if(_enableDose || _enableBioDose || _enableRBE) {
		decltype(_doseImage)* image = nullptr;
		if(_enableDose)         image = &_doseImage;
		else if(_enableBioDose) image = &_bioDoseImage;
		else if(_enableRBE)     image = &_RBEImage;

		auto currentMaterial = step->GetPreStepPoint()->GetMaterial();
		double density = currentMaterial->GetDensity();
		double mass = image->GetVoxelVolume() * density;

		deposited.dose += energyDep / mass / CLHEP::gray;
	}

	// Get information from step
	// Particle
	G4int nZ = step->GetTrack()->GetDefinition()->GetAtomicNumber();
	double kineticEnergyPerNucleon = (step->GetPreStepPoint()->GetKineticEnergy()) / (step->GetTrack()->GetDefinition()->GetAtomicMass()); //OK

	++_eventCount;

	// Accumulation of alpha/beta if ion type if known
	// -> check if the ion type is known
	if(_energyMaxForZ.count(nZ)) {
		++_eventWithKnownIonCount;

		//The max values in the database aren't being taking into account
		//so for now it's coded like this to be sure the code takes them into account
		double energyMax = _energyMaxForZ.at(nZ);

		AlphaBetaInterpolTable::const_iterator itr2;
		if (kineticEnergyPerNucleon >= energyMax) {
			// If the kinetic energy is the maximum value in the alpha beta tables,
			// we have to use the a and b coefficient for this maximum value
			Fragment fragmentKineticEnergyMax{nZ, energyMax};
			itr2 = _alphaBetaInterpolTable.find(fragmentKineticEnergyMax);
		} else {
			// We pair the ion type and the kinetic energy
			Fragment fragmentKineticEnergy{nZ, kineticEnergyPerNucleon};
			itr2 = _alphaBetaInterpolTable.upper_bound(fragmentKineticEnergy);
		}

		// Calculation of EZ, alphaDep and betaDep (K = a*EZ+b*E)
		auto const& interpol = (*itr2).second;

		double alpha = (interpol.alpha.a * kineticEnergyPerNucleon + interpol.alpha.b);
		double sqrtBeta = (interpol.sqrtBeta.a * kineticEnergyPerNucleon + interpol.sqrtBeta.b);
		
		if(alpha < 0) alpha = 0;
		if(sqrtBeta < 0) sqrtBeta = 0;

		double alphaDep = alpha * energyDep;
		double sqrtBetaDep  = sqrtBeta * energyDep;

		// Accumulate alpha/beta
		deposited.alpha     += alphaDep;
		deposited.sqrtBeta  += sqrtBetaDep;
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::ResetData() {
	if(_enableEdep)     _edepImage.Reset();
	if(_enableDose)     _doseImage.Reset();
	if(_enableBioDose)  _bioDoseImage.Reset();
	if(_enableAlphaMix) _alphaMixImage.Reset();
	if(_enableBetaMix)  _betaMixImage.Reset();
	if(_enableRBE)      _RBEImage.Reset();
}
//-----------------------------------------------------------------------------
