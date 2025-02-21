/*--
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "G4EmParameters.hh"
#include "GateBioDoseActor.hh"
#include "GateImageWithStatistic.hh"
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ios.hh>

#define GATE_BUFFERSIZE

//-----------------------------------------------------------------------------
GateBioDoseActor::GateBioDoseActor(G4String name, G4int depth):
	GateVImageActor(std::move(name), depth),
	_messenger(this)
{
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor() -- begin\n");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::Construct() {
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor -- Construct - begin\n");
	GateVImageActor::Construct();

	G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
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

		setupImage(_eventEdepImage);
		setupImage(_eventDoseImage);
		setupImage(_eventSumAlphaMixDoseImage);
		setupImage(_eventSumSqrtBetaMixDoseImage);

		setupImage(_hitEventCountImage, "hitevent_count");
		setupImage(_sumAlphaMixImage);
		setupImage(_sumSqrtBetaMixImage);
		setupImage(_sumAlphaMixDoseImage);
		setupImage(_sumSqrtBetaMixDoseImage);

		setupImage(_edepImage, "edep"); // always active
		setupImage(_doseImage); // dose output can be scaled, see SaveData()
		if(_enableDose)             setupImage(_scaledDoseImage, "dose");
		if(_enableAlphaMix)         setupImage(_alphaMixImage, "alphamix");
		if(_enableSqrtBetaMix)      setupImage(_sqrtBetaMixImage, "sqrtbetamix");
		if(_enableAlphaMixDose)     setupImage(_alphaMixDoseImage, "alphamix_dose");
		if(_enableSqrtBetaMixDose)  setupImage(_sqrtBetaMixDoseImage, "sqrtbetamix_dose");
		setupImage(_bioDoseImage, "biodose");
		if(_enableRBE)              setupImage(_rbeImage, "rbe");

		if(_enableUncertainty) {
			setupImage(_squaredDoseImage, "sq_dose");
			setupImage(_squaredAlphaMixDoseImage, "sq_alphamix_dose");
			setupImage(_squaredSqrtBetaMixDoseImage, "sq_sqrtbetamix_dose");
			setupImage(_alphaMixDoseDoseImage, "alphamixdose_dose");
			setupImage(_sqrtBetaMixDoseDoseImage, "sqrtbetamixdose_dose");
			setupImage(_alphaMixDoseSqrtBetaMixDoseImage, "alphamixdose_sqrtbetamixdose");

			setupImage(_doseUncertaintyImage, "dose_uncertainty");
			setupImage(_biodoseUncertaintyImage, "biodose_uncertainty");
			setupImage(_alphaMixUncertaintyImage, "alphamix_uncertainty");
			setupImage(_sqrtBetaMixUncertaintyImage, "sqrtbetamix_uncertainty");

			if(_enableUncertaintyDetails) {
				setupImage(_pdBiodoseAlphaMixDoseImage, "pd_biodose_alphamixdose");
				setupImage(_pdBiodoseSqrtBetaMixDoseImage, "pd_biodose_sqrtbetamixdose");
				setupImage(_varAlphaMixDoseImage, "var_alphamixdose");
				setupImage(_varSqrtBetaMixDoseImage, "var_sqrtbetamixdose");
				setupImage(_covAlphaMixDoseSqrtBetaMixDoseImage, "cov_alphamixdose_sqrtbetamixdose");
				setupImage(_covAlphaMixDoseDoseImage, "cov_alphamixdose_dose");
				setupImage(_covSqrtBetaMixDoseDoseImage, "cov_sqrtbetamixdose_dose");
			}
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
	buildDatabase();

	if(_alphaRef < 0 || _betaRef < 0)
		GateError("BioDoseActor " << GetName() << ": setAlphaRef and setBetaRef must be done");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::buildDatabase() {
	std::ifstream f(_dataBase);
	if(!f) GateError("BioDoseActor " << GetName() << ": unable to open file '" << _dataBase << "'");

	int nZ = 0;
	double prevKineticEnergy = 1;
	double prevAlpha = 1;
	double prevBeta = 1;

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

			auto alphaCoeff = interpol(prevKineticEnergy, kineticEnergy, prevAlpha, alpha);
			auto sqrtBetaCoeff = interpol(prevKineticEnergy, kineticEnergy, std::sqrt(prevBeta), std::sqrt(beta));

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
GateBioDoseActor::Coefficients GateBioDoseActor::interpol(double x1, double x2, double y1, double y2) {
	//Function for a 1D linear interpolation. It returns a pair of a and b coefficients
	double a = (y2 - y1) / (x2 - x1);
	double b = y1 - x1 * a;
	return {a, b};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::BeginOfRunAction(const G4Run* r) {
	GateVActor::BeginOfRunAction(r);
	GateDebugMessage("Actor", 3, "GateBioDoseActor -- Begin of Run\n");
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
	_eventSumAlphaMixDoseImage.Reset();
	_eventSumSqrtBetaMixDoseImage.Reset();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::EndOfEventAction(const G4Event* e) {
	GateVActor::EndOfEventAction(e);

	for(auto const& index: _eventVoxelIndices) {
		auto const eventEdep = _eventEdepImage.GetValue(index);
		auto const eventDose = _eventDoseImage.GetValue(index);
		auto const eventSumAlphaMixDose = _eventSumAlphaMixDoseImage.GetValue(index);
		auto const eventSumSqrtBetaMixDose = _eventSumSqrtBetaMixDoseImage.GetValue(index);

		_voxelIndices.insert(index);
		_hitEventCountImage.AddValue(index, 1);

		_edepImage.AddValue(index, eventEdep);
		_doseImage.AddValue(index, eventDose);
		_sumAlphaMixDoseImage.AddValue(index, eventSumAlphaMixDose);
		_sumSqrtBetaMixDoseImage.AddValue(index, eventSumSqrtBetaMixDose);

		if(_enableUncertainty) { // TODO use event hit count to have a mean of alpha/sqrt(beta) instead of sum?
			_squaredDoseImage.AddValue(index, eventDose * eventDose);
			_squaredAlphaMixDoseImage.AddValue(index, eventSumAlphaMixDose * eventSumAlphaMixDose);
			_squaredSqrtBetaMixDoseImage.AddValue(index, eventSumSqrtBetaMixDose * eventSumSqrtBetaMixDose);

			_alphaMixDoseDoseImage.AddValue(index, eventSumAlphaMixDose * eventDose);
			_sqrtBetaMixDoseDoseImage.AddValue(index, eventSumSqrtBetaMixDose * eventDose);
			_alphaMixDoseSqrtBetaMixDoseImage.AddValue(index, eventSumAlphaMixDose * eventSumSqrtBetaMixDose);
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

	// Accumulate energy inconditionnaly
	_eventEdepImage.AddValue(index, energyDep);

	auto* currentMaterial = step->GetPreStepPoint()->GetMaterial();
	double density = currentMaterial->GetDensity();
	double mass = _bioDoseImage.GetVoxelVolume() * density;
	double dose = energyDep / mass / CLHEP::gray;

	_eventDoseImage.AddValue(index, dose);

	// Get information from step
	// Particle
	G4int nZ = step->GetTrack()->GetDefinition()->GetAtomicNumber();
	double kineticEnergyPerNucleon = (step->GetPreStepPoint()->GetKineticEnergy()) / (step->GetTrack()->GetDefinition()->GetAtomicMass());

	++_stepCount;

	// Accumulation of weighted alpha/sqrt(beta) if ion type if known
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

		double alpha = interpol.alpha.a * kineticEnergyPerNucleon + interpol.alpha.b;
		double sqrtBeta = interpol.sqrtBeta.a * kineticEnergyPerNucleon + interpol.sqrtBeta.b;

		if(alpha < 0) alpha = 0;
		if(sqrtBeta < 0) sqrtBeta = 0;

		_sumAlphaMixImage.AddValue(index, alpha * energyDep);
		_sumSqrtBetaMixImage.AddValue(index, sqrtBeta * energyDep);

		// Accumulate weighted alpha/sqrt(beta)
		_eventSumAlphaMixDoseImage.AddValue(index, alpha * dose);
		_eventSumSqrtBetaMixDoseImage.AddValue(index, sqrtBeta * dose);

		_eventVoxelIndices.insert(index);
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::updateData() {
	auto const sqAlphaRef = _alphaRef * _alphaRef;
	double const n = _currentEvent;

	for(auto const& index: _voxelIndices) {
		auto const hitEventCount = _hitEventCountImage.GetValue(index);

		auto const alphaMix = _sumAlphaMixImage.GetValue(index) / _edepImage.GetValue(index);
		auto const sqrtBetaMix = _sumSqrtBetaMixImage.GetValue(index) / _edepImage.GetValue(index);

		auto const alphaMixDose = _sumAlphaMixDoseImage.GetValue(index);
		auto const sqrtBetaMixDose = _sumSqrtBetaMixDoseImage.GetValue(index);

		auto const dose = _doseImage.GetValue(index);
		auto const scaledDose = _doseScaleFactor * dose;
		auto const sqScaledDose = scaledDose * scaledDose;
		auto const delta = sqAlphaRef + 4 * _betaRef * (alphaMixDose + sqrtBetaMixDose * sqrtBetaMixDose);

		double sqrtDelta = 0;
		if(delta >= 0)
			sqrtDelta = std::sqrt(delta);

		// Calculate biological dose and RBE
		double biodose  = 0;
		double rbe      = 0;

		if(scaledDose > 0 && alphaMixDose != 0 && sqrtBetaMixDose != 0)
			biodose = (-_alphaRef + sqrtDelta) / (2 * _betaRef);
		if(biodose < 0) biodose = 0; // TODO improve

		if(scaledDose > 0)
			rbe = biodose / scaledDose;

		if(_enableUncertainty) {
			if(scaledDose > 0 && alphaMixDose != 0 && sqrtBetaMixDose != 0 && sqrtDelta > 0 && _currentEvent > 0) {
				auto var = [](double n, double sq, double v) {
					return 1 / (n - 1) * (sq - v * v);
				};

				auto cov = [](double n, double ab, double a, double b) {
					return 1 / (n - 1) * (ab - a * b);
				};

				auto const squaredAlphaMixDose = _squaredAlphaMixDoseImage.GetValue(index);
				auto const squaredSqrtBetaMixDose = _squaredSqrtBetaMixDoseImage.GetValue(index);
				auto const squaredDose = _squaredDoseImage.GetValue(index);

				auto const pdBiodoseAlphaMixDose = 1 / sqrtDelta;
				auto const pdBiodoseSqrtBetaMixDose = 2 * sqrtBetaMixDose / sqrtDelta;

				auto const varAlphaMixDose = var(n, squaredAlphaMixDose, alphaMixDose);
				auto const varSqrtBetaMixDose = var(n, squaredSqrtBetaMixDose, sqrtBetaMixDose);
				auto const varDose = var(n, squaredDose, dose);

				auto const alphaMixDoseSqrtBetaMixDose = _alphaMixDoseSqrtBetaMixDoseImage.GetValue(index);
				auto const covAlphaMixDoseSqrtBetaMixDose = cov(n, alphaMixDoseSqrtBetaMixDose, alphaMixDose, sqrtBetaMixDose);

				auto const alphaMixDoseDose = _alphaMixDoseDoseImage.GetValue(index);
				auto const covAlphaMixDoseDose = cov(n, alphaMixDoseDose, alphaMixDose, dose);

				auto const sqrtBetaMixDoseDose = _sqrtBetaMixDoseDoseImage.GetValue(index);
				auto const covSqrtBetaMixDoseDose = cov(n, sqrtBetaMixDoseDose, sqrtBetaMixDose, dose);

				// TODO mistake here, must square lonely pd
				auto const partAlphaMixDose = pdBiodoseAlphaMixDose * pdBiodoseAlphaMixDose * varAlphaMixDose;
				auto const partSqrtBetaMixDose = pdBiodoseSqrtBetaMixDose * pdBiodoseSqrtBetaMixDose * varSqrtBetaMixDose;
				auto const partAlphaMixDoseSqrtBetaMixDose = 2 * pdBiodoseAlphaMixDose * pdBiodoseSqrtBetaMixDose * covAlphaMixDoseSqrtBetaMixDose;
				auto const varBiodose = partAlphaMixDose + partSqrtBetaMixDose + partAlphaMixDoseSqrtBetaMixDose;

				auto const varAlphaMixDosePartA = (1 / dose / dose) * varAlphaMixDose;
				auto const varAlphaMixDosePartB = (alphaMixDose * alphaMixDose / dose / dose / dose / dose) * varDose;
				auto const varAlphaMixDosePartC = 2 * (alphaMixDose / dose / dose / dose) * covAlphaMixDoseDose;
				auto const varAlphaMix = varAlphaMixDosePartA + varAlphaMixDosePartB + varAlphaMixDosePartC;

				auto const varSqrtBetaMixDosePartA = (1 / dose / dose) * varSqrtBetaMixDose;
				auto const varSqrtBetaMixDosePartB = (sqrtBetaMixDose * sqrtBetaMixDose / dose / dose / dose / dose) * varDose;
				auto const varSqrtBetaMixDosePartC = 2 * (sqrtBetaMixDose / dose / dose / dose) * covSqrtBetaMixDoseDose;
				auto const varSqrtBetaMix = varSqrtBetaMixDosePartA + varSqrtBetaMixDosePartB + varSqrtBetaMixDosePartC;

				auto uncertaintyDose = std::sqrt(varDose) / dose;
				auto uncertaintyBiodose = std::sqrt(varBiodose) / biodose;
				auto uncertaintyAlphaMix = std::sqrt(varAlphaMix) / alphaMix;
				auto uncertaintySqrtBetaMix = std::sqrt(varSqrtBetaMix) / sqrtBetaMix;

				_doseUncertaintyImage.SetValue(index, uncertaintyDose);
				_biodoseUncertaintyImage.SetValue(index, uncertaintyBiodose);
				_alphaMixUncertaintyImage.SetValue(index, uncertaintyAlphaMix);
				_sqrtBetaMixUncertaintyImage.SetValue(index, uncertaintySqrtBetaMix);

				if(_enableUncertaintyDetails) {
					_pdBiodoseAlphaMixDoseImage.SetValue(index, pdBiodoseAlphaMixDose);
					_pdBiodoseSqrtBetaMixDoseImage.SetValue(index, pdBiodoseSqrtBetaMixDose);
					_varAlphaMixDoseImage.SetValue(index, varAlphaMixDose);
					_varSqrtBetaMixDoseImage.SetValue(index, varSqrtBetaMixDose);
					_covAlphaMixDoseSqrtBetaMixDoseImage.SetValue(index, covAlphaMixDoseSqrtBetaMixDose);
					_covAlphaMixDoseDoseImage.SetValue(index, covAlphaMixDoseDose);
					_covSqrtBetaMixDoseDoseImage.SetValue(index, covSqrtBetaMixDoseDose);
				}
			} else {
				_doseUncertaintyImage.SetValue(index, 1);
				_biodoseUncertaintyImage.SetValue(index, 1);
				_alphaMixUncertaintyImage.SetValue(index, 1);
				_sqrtBetaMixUncertaintyImage.SetValue(index, 1);

				if(_enableUncertaintyDetails) {
					_pdBiodoseAlphaMixDoseImage.SetValue(index, 1);
					_pdBiodoseSqrtBetaMixDoseImage.SetValue(index, 1);
					_varAlphaMixDoseImage.SetValue(index, 1);
					_varSqrtBetaMixDoseImage.SetValue(index, 1);
					_covAlphaMixDoseSqrtBetaMixDoseImage.SetValue(index, 1);
					_covAlphaMixDoseDoseImage.SetValue(index, 1);
					_covSqrtBetaMixDoseDoseImage.SetValue(index, 1);
				}
			}
		}

		// Write data
		if(_enableDose)             _scaledDoseImage.SetValue(index, scaledDose);
		if(_enableAlphaMix)         _alphaMixImage.SetValue(index, alphaMix);
		if(_enableSqrtBetaMix)      _sqrtBetaMixImage.SetValue(index, sqrtBetaMix);
		if(_enableAlphaMixDose)     _alphaMixDoseImage.SetValue(index, alphaMixDose);
		if(_enableSqrtBetaMixDose)  _sqrtBetaMixDoseImage.SetValue(index, sqrtBetaMixDose);
		_bioDoseImage.SetValue(index, biodose);
		if(_enableRBE)              _rbeImage.SetValue(index, rbe);
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::SaveData() {
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor::SaveData() known ion events / total events: " << _eventWithKnownIonCount << " / " << _eventCount << "\n");

	updateData();

	GateVActor::SaveData();

	if(_enableEdep)             _edepImage.SaveData(_currentEvent);
	if(_enableDose)             _scaledDoseImage.SaveData(_currentEvent);
	if(_enableAlphaMix)         _alphaMixImage.SaveData(_currentEvent);
	if(_enableSqrtBetaMix)      _sqrtBetaMixImage.SaveData(_currentEvent);
	if(_enableAlphaMixDose)     _alphaMixDoseImage.SaveData(_currentEvent);
	if(_enableSqrtBetaMixDose)  _sqrtBetaMixDoseImage.SaveData(_currentEvent);
	_bioDoseImage.SaveData(_currentEvent);
	if(_enableRBE)              _rbeImage.SaveData(_currentEvent);
	if(_enableUncertainty) {
		_doseUncertaintyImage.SaveData(_currentEvent);
	  _biodoseUncertaintyImage.SaveData(_currentEvent);
		_alphaMixUncertaintyImage.SaveData(_currentEvent);
		_sqrtBetaMixUncertaintyImage.SaveData(_currentEvent);

		if(_enableUncertaintyDetails) {
			_squaredDoseImage.SaveData(_currentEvent);
			_squaredAlphaMixDoseImage.SaveData(_currentEvent);
			_squaredSqrtBetaMixDoseImage.SaveData(_currentEvent);
			_alphaMixDoseDoseImage.SaveData(_currentEvent);
			_sqrtBetaMixDoseDoseImage.SaveData(_currentEvent);
			_alphaMixDoseSqrtBetaMixDoseImage.SaveData(_currentEvent);

			_pdBiodoseAlphaMixDoseImage.SaveData(_currentEvent);
			_pdBiodoseSqrtBetaMixDoseImage.SaveData(_currentEvent);
			_varAlphaMixDoseImage.SaveData(_currentEvent);
			_varSqrtBetaMixDoseImage.SaveData(_currentEvent);
			_covAlphaMixDoseSqrtBetaMixDoseImage.SaveData(_currentEvent);
			_covAlphaMixDoseDoseImage.SaveData(_currentEvent);
			_covSqrtBetaMixDoseDoseImage.SaveData(_currentEvent);
		}
	}
	if(_enableHitEventCount)  _hitEventCountImage.SaveData(_currentEvent);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::ResetData() {
	_eventVoxelIndices.clear();
	_voxelIndices.clear();

	_eventEdepImage.Reset();
	_eventDoseImage.Reset();
	_eventSumAlphaMixDoseImage.Reset();
	_eventSumSqrtBetaMixDoseImage.Reset();

	_hitEventCountImage.Reset();
	_sumAlphaMixImage.Reset();
	_sumSqrtBetaMixImage.Reset();
	_sumAlphaMixDoseImage.Reset();
	_sumSqrtBetaMixDoseImage.Reset();

	if(_enableEdep)             _edepImage.Reset();
	_doseImage.Reset();
	if(_enableDose)             _scaledDoseImage.Reset();
	if(_enableAlphaMix)         _alphaMixImage.Reset();
	if(_enableSqrtBetaMix)      _sqrtBetaMixImage.Reset();
	if(_enableAlphaMixDose)     _alphaMixDoseImage.Reset();
	if(_enableSqrtBetaMixDose)  _sqrtBetaMixDoseImage.Reset();
	_bioDoseImage.Reset();
	if(_enableRBE)              _rbeImage.Reset();

	if(_enableUncertainty) {
		_squaredDoseImage.Reset();
		_squaredAlphaMixDoseImage.Reset();
		_squaredSqrtBetaMixDoseImage.Reset();
		_alphaMixDoseDoseImage.Reset();
		_sqrtBetaMixDoseDoseImage.Reset();
		_alphaMixDoseSqrtBetaMixDoseImage.Reset();

		_doseUncertaintyImage.Reset();
		_biodoseUncertaintyImage.Reset();
		_alphaMixUncertaintyImage.Reset();
		_sqrtBetaMixUncertaintyImage.Reset();

		if(_enableUncertaintyDetails) {
			_pdBiodoseAlphaMixDoseImage.Reset();
			_pdBiodoseSqrtBetaMixDoseImage.Reset();
			_varAlphaMixDoseImage.Reset();
			_varSqrtBetaMixDoseImage.Reset();
			_covAlphaMixDoseSqrtBetaMixDoseImage.Reset();
			_covAlphaMixDoseDoseImage.Reset();
			_covSqrtBetaMixDoseDoseImage.Reset();
		}
	}
}
//-----------------------------------------------------------------------------
