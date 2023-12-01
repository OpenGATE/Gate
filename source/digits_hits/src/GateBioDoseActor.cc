/*--
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
	GateVImageActor(std::move(name), depth),
	_currentEvent(0),
	_messenger(this),
	_alphaRef(-1),
	_betaRef(-1),
	_sobpWeight(0),
	_enableEdep(false),
	_enableDose(true),
	_enableBioDose(true),
	_enableAlphaMix(false),
	_enableSqrtBetaMix(false),
	_enableRBE(false),
	_enableUncertainties(true)
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
	EnableEndOfEventAction(true);
	EnablePreUserTrackingAction(false);
	EnablePostUserTrackingAction(false);
	EnableUserSteppingAction(true);

	// Outputs
	{
		G4String basename = removeExtension(mSaveFilename);
		G4String ext = getExtension(mSaveFilename);

		auto setupImage = [&](GateImageWithStatistic& image, std::string const& suffix = "") {
			SetOriginTransformAndFlagToImage(image);
			image.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
			image.Allocate();

			if(!suffix.empty()) {
				G4String filename = basename + "_" + suffix + "." + ext;
				image.SetFilename(filename);
			}
		};

		setupImage(_eventCountImage);

		if(_enableEdep)         setupImage(_edepImage, "edep");
		if(_enableDose)         setupImage(_doseImage, "dose");
		setupImage(_bioDoseImage, "biodose");
		setupImage(_alphaMixImage, "alphamix");
		setupImage(_sqrtBetaMixImage, "sqrtbetamix");
		if(_enableRBE)          setupImage(_rbeImage, "rbe");

		setupImage(_eventEdepImage);
		setupImage(_eventDoseImage);
		setupImage(_eventAlphaImage);
		setupImage(_eventSqrtBetaImage);

		if(_enableUncertainties) {
			setupImage(_biodoseUncertaintyImage, "biodose_uncertainty");

			setupImage(_squaredDoseImage);
			setupImage(_squaredAlphaMixImage);
			setupImage(_squaredSqrtBetaMixImage);

			setupImage(_alphaMixSqrtBetaMixImage);
			setupImage(_alphaMixDoseImage);
			setupImage(_sqrtBetaMixDoseImage);
		}
	}

	ResetData();

	///////////////////////////////////////////////////////////////////////////////////////////
	//Just matrix information
	G4cout << "Memory space to store physical dose into " << mResolution.x() * mResolution.y() * mResolution.z() << " voxels has been allocated " << G4endl;

	// SOBP
	if(_sobpWeight == 0) { _sobpWeight = 1; }

	//Building the cell line information
	_dataBase = "data/" + _cellLine + "_" + _bioPhysicalModel + ".db";
	BuildDatabase();

	if(_alphaRef < 0 || _betaRef < 0)
		GateError("BioDoseActor " << GetName() << ": setAlphaRef and setBetaRef must be done");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::BuildDatabase() {
	std::ifstream f(_dataBase);
	if(!f) GateError("BioDoseActor " << GetName() << ": unable to open file '" << _dataBase << "'");

	int nZ = 0;
	double prevKineticEnergy = 1;
	double prevAlpha = 1;
	double prevBeta =1;

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
			double kineticEnergy = 0;
			double alpha = 0;
			double beta = 0;
			std::istringstream{firstCol} >> kineticEnergy;
			iss >> alpha;
			iss >> beta;

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

	/* ******************* */
	// TODO remove
	G4String basename = removeExtension(mSaveFilename);
	G4String ext = getExtension(mSaveFilename);
	auto setupImage = [&](GateImageWithStatistic& image, std::string const& suffix = "") {
		SetOriginTransformAndFlagToImage(image);
		image.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
		image.Allocate();

		if(!suffix.empty()) {
			G4String filename = basename + "_" + suffix + "." + ext;
			image.SetFilename(filename);
		}
	};

	GateImageWithStatistic partAlphaMixImg;
	GateImageWithStatistic partSqrtBetaMixImg;
	GateImageWithStatistic partDoseImg;
	GateImageWithStatistic partAlphaMixSqrtBetaMixImg;
	GateImageWithStatistic partAlphaMixDoseImg;
	GateImageWithStatistic partSqrtBetaMixDoseImg;
	GateImageWithStatistic partSquaredBioDoseImg;

	GateImageWithStatistic squaredPdBioDoseAlphaMixImg;
	GateImageWithStatistic squaredPdBioDoseSqrtBetaMixImg;
	GateImageWithStatistic squaredPdBioDoseDoseImg;
	GateImageWithStatistic varAlphaMixImg;
	GateImageWithStatistic varSqrtBetaMixImg;
	GateImageWithStatistic varDoseImg;

	setupImage(partAlphaMixImg, "part_alphamix");
	setupImage(partSqrtBetaMixImg, "part_sqrtbetamix");
	setupImage(partDoseImg, "part_dose");
	setupImage(partAlphaMixSqrtBetaMixImg, "part_alphamixsqrtbetamix");
	setupImage(partAlphaMixDoseImg, "part_alphamixdose");
	setupImage(partSqrtBetaMixDoseImg, "part_sqrtbetamixdose");
	setupImage(partSquaredBioDoseImg, "part_squaredbiodose");

	setupImage(squaredPdBioDoseAlphaMixImg, "squared_pdbiodosealphamix");
	setupImage(squaredPdBioDoseSqrtBetaMixImg, "squared_pdbiodosesqrtbetamix");
	setupImage(squaredPdBioDoseDoseImg, "squared_pdbiodosedose");
	setupImage(varAlphaMixImg, "var_alphamix");
	setupImage(varSqrtBetaMixImg, "var_sqrtbetamix");
	setupImage(varDoseImg, "var_dose");
	/* ******************* */

	auto const sqAlphaRef = _alphaRef * _alphaRef;
	double const n = _currentEvent;

	for(auto const& [index, deposited]: _depositedMap) {
		auto const eventCount = _eventCountImage.GetValue(index);

		auto const alphaMixMean = _alphaMixImage.GetValue(index) / eventCount;
		auto const sqrtBetaMixMean = _sqrtBetaMixImage.GetValue(index) / eventCount;
		auto const dose = deposited.dose;
		auto const scaledDose = _doseScaleFactor * dose;
		auto const sqScaledDose = scaledDose * scaledDose;
		auto const delta = sqAlphaRef + 4 * _betaRef *
			(alphaMixMean * scaledDose + sqrtBetaMixMean * sqrtBetaMixMean * sqScaledDose);

		double sqrtDelta = 0;
		if(delta >= 0)
			sqrtDelta = std::sqrt(delta);

		// Calculate biological dose and RBE
		double biodose  = 0;
		double rbe      = 0;

		if(scaledDose > 0 && alphaMixMean != 0 && sqrtBetaMixMean != 0)
			biodose = (-_alphaRef + sqrtDelta) / (2 * _betaRef);
		if(biodose < 0) biodose = 0; // TODO improve

		if(scaledDose > 0)
			rbe = biodose / scaledDose;

		if(index == 80) { // TODO remove
			std::ofstream of{"/tmp/eoea", std::ios_base::app};
			of << "alphaRef: " << _alphaRef << '\n';
			of << "sqrtBetaRef: " << _betaRef << '\n';
			of << "s: " << _doseScaleFactor << '\n';
			of << "eventCount: " << eventCount << '\n';
			of << "alphaMixMean: " << _alphaMixImage.GetValue(index) << " / " << eventCount << " = " << alphaMixMean << '\n';
			of << "sqrtBetaMixMean: " << sqrtBetaMixMean << '\n';
			of << "delta: " << delta << '\n';
			of << "final sum dose: " << dose << '\n';
			of << "final sum squared dose: " << _squaredDoseImage.GetValue(index) << '\n';
			of << "biodose: " << biodose << '\n';
			of << "rbe: " << rbe << '\n';
		}

		if(_enableUncertainties) {
			if(scaledDose > 0 && alphaMixMean != 0 && sqrtBetaMixMean != 0 && sqrtDelta > 0 && _currentEvent > 0) {
				double const doseMean = dose / n;

				double sumSquaredAlphaMix = _squaredAlphaMixImage.GetValue(index);
				double sumSquaredSqrtBetaMix = _squaredSqrtBetaMixImage.GetValue(index);
				double sumSquaredDose = _squaredDoseImage.GetValue(index);

				double pdBiodoseAlphaMix = scaledDose / sqrtDelta;
				double pdBiodoseSqrtBetaMix = 2 * sqScaledDose * sqrtBetaMixMean / sqrtDelta;
				double pdBiodoseDose = (
					(alphaMixMean * _doseScaleFactor) +
					2 * sqrtBetaMixMean * sqrtBetaMixMean * _doseScaleFactor * scaledDose
				) / sqrtDelta;

				double varAlphaMix = sumSquaredAlphaMix / eventCount - alphaMixMean * alphaMixMean;
				double varSqrtBetaMix = sumSquaredSqrtBetaMix / eventCount - sqrtBetaMixMean * sqrtBetaMixMean;
				double varDose = sumSquaredDose / n - doseMean * doseMean;

				double sumAlphaMixSqrtBetaMix = _alphaMixSqrtBetaMixImage.GetValue(index);
				double sumAlphaMixDose = _alphaMixDoseImage.GetValue(index);
				double sumSqrtBetaMixDose = _sqrtBetaMixDoseImage.GetValue(index);
				double covAlphaMixSqrtBetaMix = sumAlphaMixSqrtBetaMix / eventCount - alphaMixMean * sqrtBetaMixMean;
				double covAlphaMixDose = sumAlphaMixDose / n - alphaMixMean * doseMean;
				double covSqrtBetaMixDose = sumSqrtBetaMixDose / n - sqrtBetaMixMean * doseMean;

				double partAlphaMix = pdBiodoseAlphaMix * pdBiodoseAlphaMix * varAlphaMix;
				double partSqrtBetaMix = pdBiodoseSqrtBetaMix * pdBiodoseSqrtBetaMix * varSqrtBetaMix;
				double partDose = pdBiodoseDose * pdBiodoseDose * varDose;
				double partAlphaMixSqrtBetaMix = 2 * pdBiodoseAlphaMix * pdBiodoseSqrtBetaMix * covAlphaMixSqrtBetaMix;
				double partAlphaMixDose = 2 * pdBiodoseAlphaMix * pdBiodoseDose * covAlphaMixDose;
				double partSqrtBetaMixDose = 2 * pdBiodoseSqrtBetaMix * pdBiodoseDose * covSqrtBetaMixDose;

				{ // TODO remove
					partAlphaMixImg.SetValue(index, partAlphaMix);
					partSqrtBetaMixImg.SetValue(index, partSqrtBetaMix);
					partDoseImg.SetValue(index, partDose);
					partAlphaMixDoseImg.SetValue(index, partAlphaMixDose);
					partAlphaMixSqrtBetaMixImg.SetValue(index, partAlphaMixSqrtBetaMix);
					partSqrtBetaMixDoseImg.SetValue(index, partSqrtBetaMixDose);
					partSquaredBioDoseImg.SetValue(index, biodose * biodose);

					squaredPdBioDoseAlphaMixImg.SetValue(index, pdBiodoseAlphaMix * pdBiodoseAlphaMix);
					squaredPdBioDoseSqrtBetaMixImg.SetValue(index, pdBiodoseSqrtBetaMix * pdBiodoseSqrtBetaMix);
					squaredPdBioDoseDoseImg.SetValue(index, pdBiodoseDose * pdBiodoseDose);
					varAlphaMixImg.SetValue(index, varAlphaMix);
					varSqrtBetaMixImg.SetValue(index, varSqrtBetaMix);
					varDoseImg.SetValue(index, varDose);
				}

				double uncertaintyBiodose = std::sqrt(
					partAlphaMix + partSqrtBetaMix + partDose +
					partAlphaMixSqrtBetaMix + partAlphaMixDose + partSqrtBetaMixDose
				) / biodose;

				_biodoseUncertaintyImage.SetValue(index, uncertaintyBiodose);
			} else {
				_biodoseUncertaintyImage.SetValue(index, 1);
			}
		}

		// Write data
		if(_enableEdep)         _edepImage.SetValue(index, deposited.energy);
		if(_enableDose)         _doseImage.SetValue(index, scaledDose);
		_bioDoseImage.SetValue(index, biodose);
		if(_enableRBE)          _rbeImage.SetValue(index, rbe);
	}

	GateVActor::SaveData();

	if(_enableEdep)           _edepImage.SaveData(_currentEvent);
	if(_enableDose)           _doseImage.SaveData(_currentEvent);
	if(_enableBioDose)        _bioDoseImage.SaveData(_currentEvent);
	if(_enableAlphaMix)       _alphaMixImage.SaveData(_currentEvent);
	if(_enableSqrtBetaMix)    _sqrtBetaMixImage.SaveData(_currentEvent);
	if(_enableRBE)            _rbeImage.SaveData(_currentEvent);
	if(_enableUncertainties)  _biodoseUncertaintyImage.SaveData(_currentEvent);

	{ // TODO remove
		partAlphaMixImg.SaveData(_currentEvent);
		partSqrtBetaMixImg.SaveData(_currentEvent);
		partDoseImg.SaveData(_currentEvent);
		partAlphaMixDoseImg.SaveData(_currentEvent);
		partAlphaMixSqrtBetaMixImg.SaveData(_currentEvent);
		partSqrtBetaMixDoseImg.SaveData(_currentEvent);
		partSquaredBioDoseImg.SaveData(_currentEvent);

		squaredPdBioDoseAlphaMixImg.SaveData(_currentEvent);
		squaredPdBioDoseSqrtBetaMixImg.SaveData(_currentEvent);
		squaredPdBioDoseDoseImg.SaveData(_currentEvent);
		varAlphaMixImg.SaveData(_currentEvent);
		varSqrtBetaMixImg.SaveData(_currentEvent);
		varDoseImg.SaveData(_currentEvent);
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::BeginOfRunAction(const G4Run* r) {
	GateVActor::BeginOfRunAction(r);
	GateDebugMessage("Actor", 3, "GateBioDoseActor -- Begin of Run\n");

	ResetData();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::EndOfRunAction(const G4Run* r) {
	GateVActor::EndOfRunAction(r);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::BeginOfEventAction(const G4Event* e) {
	GateVActor::BeginOfEventAction(e);
	++_currentEvent;

	_eventVoxelIndices.clear();

	_eventEdepImage.Reset();
	_eventDoseImage.Reset();
	_eventAlphaImage.Reset();
	_eventSqrtBetaImage.Reset();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::EndOfEventAction(const G4Event* e) {
	GateVActor::EndOfEventAction(e);

	for(auto const& index: _eventVoxelIndices) {
		auto const eventEdep = _eventEdepImage.GetValue(index);
		auto const eventDose = _eventDoseImage.GetValue(index);
		auto const eventAlphaMix = _eventAlphaImage.GetValue(index) / eventEdep;
		auto const eventSqrtBetaMix = _eventSqrtBetaImage.GetValue(index) / eventEdep;

		_eventCountImage.AddValue(index, 1);

		_alphaMixImage.AddValue(index, eventAlphaMix);
		_sqrtBetaMixImage.AddValue(index, eventSqrtBetaMix);

		if(_enableUncertainties) {
			_squaredDoseImage.AddValue(index, eventDose * eventDose);
			_squaredAlphaMixImage.AddValue(index, eventAlphaMix * eventAlphaMix);
			_squaredSqrtBetaMixImage.AddValue(index, eventSqrtBetaMix * eventSqrtBetaMix);

			_alphaMixSqrtBetaMixImage.AddValue(index, eventAlphaMix * eventSqrtBetaMix);
			_alphaMixDoseImage.AddValue(index, eventAlphaMix * eventDose);
			_sqrtBetaMixDoseImage.AddValue(index, eventSqrtBetaMix * eventDose);
		}

		if(index == 80) { // TODO remove
			std::ofstream of{"/tmp/alpha", std::ios_base::app};
			of << "EVENT alpha: " << _eventAlphaImage.GetValue(index) << '\n';
			of << "EVENT sqrtBeta: " << _eventSqrtBetaImage.GetValue(index) << '\n';
			of << "EVENT energyDep: " << eventEdep << '\n';
			of << "--------------------\n";
		}

		std::ofstream of{"/tmp/check", std::ios_base::app};
		of << "index: " << index << '\n';
		if(index == 80 || index == 320) {
			std::ofstream of{"/tmp/eoea", std::ios_base::app};
			of << "index: " << index << '\n';
			of << "eventEdep: " << eventEdep << '\n';
			of << "eventDose: " << eventDose << '\n';
			of << "eventDose²: " << eventDose * eventDose << '\n';
			of << "current sum dose: " << _depositedMap.at(index).dose << '\n';
			of << "current sum squared dose: " << _squaredDoseImage.GetValue(index) << '\n';
			of << "------------------------\n";
		}
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
	double const weight     = step->GetTrack()->GetWeight();
	double const energyDep  = step->GetTotalEnergyDeposit() * weight;

	if(energyDep == 0)  return;
	if(index < 0)       return;

	auto it = _depositedMap.find(index);
	if(it == std::end(_depositedMap)) {
		_depositedMap[index] = {0, 0, 0, 0};
		it = _depositedMap.find(index);
	}

	auto& deposited = (*it).second;

	// Accumulate energy inconditionnaly
	deposited.energy += energyDep;

	if(_enableDose || _enableBioDose || _enableRBE) {
		auto* currentMaterial = step->GetPreStepPoint()->GetMaterial();
		double density = currentMaterial->GetDensity();
		double mass = _bioDoseImage.GetVoxelVolume() * density;
		double dose = energyDep / mass / CLHEP::gray;

		deposited.dose += dose;
		_eventDoseImage.AddValue(index, dose);
	}

	// Get information from step
	// Particle
	G4int nZ = step->GetTrack()->GetDefinition()->GetAtomicNumber();
	double kineticEnergyPerNucleon = (step->GetPreStepPoint()->GetKineticEnergy()) / (step->GetTrack()->GetDefinition()->GetAtomicMass());

	++_stepCount;

	// Accumulation of alpha/beta if ion type if known
	// -> check if the ion type is known
	if(_energyMaxForZ.count(nZ) != 0) {
		++_stepWithKnownIonCount;

		double energyMax = _energyMaxForZ.at(nZ);

		AlphaBetaInterpolTable::const_iterator itInterpol;
		if(kineticEnergyPerNucleon >= energyMax) {
			Fragment fragmentKineticEnergyMax{nZ, energyMax};
			itInterpol = _alphaBetaInterpolTable.find(fragmentKineticEnergyMax);
		} else {
			Fragment fragmentKineticEnergy{nZ, kineticEnergyPerNucleon};
			itInterpol = _alphaBetaInterpolTable.upper_bound(fragmentKineticEnergy);
		}

		// Calculation of alphaDep and betaDep (K = (a*Z+b)*E)
		auto const& interpol = (*itInterpol).second;

		double alpha = (interpol.alpha.a * kineticEnergyPerNucleon + interpol.alpha.b) * energyDep;
		double sqrtBeta = (interpol.sqrtBeta.a * kineticEnergyPerNucleon + interpol.sqrtBeta.b) * energyDep;

		if(index == 80) { // TODO remove
			std::ofstream of{"/tmp/alpha", std::ios_base::app};
			of << "alpha: " << (interpol.alpha.a * kineticEnergyPerNucleon + interpol.alpha.b) << '\n';
			of << "sqrtBeta: " << (interpol.sqrtBeta.a * kineticEnergyPerNucleon + interpol.sqrtBeta.b) << '\n';
			of << "energyDep: " << energyDep << '\n';
			of << "--------------------\n";
		}

		if(alpha < 0) alpha = 0;
		if(sqrtBeta < 0) sqrtBeta = 0;

		// Accumulate alpha/beta
		deposited.alpha     += alpha;
		deposited.sqrtBeta  += sqrtBeta;

		_eventEdepImage.AddValue(index, energyDep);
		_eventAlphaImage.AddValue(index, alpha);
		_eventSqrtBetaImage.AddValue(index, sqrtBeta);

		_eventVoxelIndices.insert(index);
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::ResetData() {
	_eventCountImage.Reset();

	if(_enableEdep)         _edepImage.Reset();
	if(_enableDose)         _doseImage.Reset();
	_bioDoseImage.Reset();
	_alphaMixImage.Reset();
	_sqrtBetaMixImage.Reset();
	if(_enableRBE)          _rbeImage.Reset();

	_eventEdepImage.Reset();
	_eventDoseImage.Reset();
	_eventAlphaImage.Reset();
	_eventSqrtBetaImage.Reset();

	if(_enableUncertainties) {
		_biodoseUncertaintyImage.Reset();

		_squaredDoseImage.Reset();
		_squaredAlphaMixImage.Reset();
		_squaredSqrtBetaMixImage.Reset();

		_alphaMixSqrtBetaMixImage.Reset();
		_alphaMixDoseImage.Reset();
		_sqrtBetaMixDoseImage.Reset();
	}
}
//-----------------------------------------------------------------------------
