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

		setupImage(_hitEventCountImage, "hitevent_count");

		setupImage(_eventEdepImage);
		setupImage(_eventDoseImage);
		setupImage(_eventAlphaImage);
		setupImage(_eventSqrtBetaImage);

		if(_enableEdep)         setupImage(_edepImage, "edep");
		setupImage(_doseImage); // dose output can be scaled, see SaveData()
		if(_enableDose) setupImage(_scaledDoseImage, "dose");
		setupImage(_alphaMixImage, "alphamix");
		setupImage(_sqrtBetaMixImage, "sqrtbetamix");
		setupImage(_bioDoseImage, "biodose");
		if(_enableRBE)          setupImage(_rbeImage, "rbe");

		if(_enableUncertainty) {
			setupImage(_doseUncertaintyImage, "dose_uncertainty");
			setupImage(_biodoseUncertaintyImage, "biodose_uncertainty");
			setupImage(_squaredDoseImage);
			setupImage(_squaredAlphaMixImage);
			setupImage(_squaredSqrtBetaMixImage);
			setupImage(_alphaMixSqrtBetaMixImage);
			setupImage(_alphaMixDoseImage);
			setupImage(_sqrtBetaMixDoseImage);

			if(_enableUncertaintyDetails) {
				setupImage(_pdBiodoseAlphaMixMeanImage, "pd_biodose_alphamixmean");
				setupImage(_pdBiodoseSqrtBetaMixMeanImage, "pd_biodose_sqrtbetamixmean");
				setupImage(_pdBiodoseDoseMeanImage, "pd_biodose_dosemean");
				setupImage(_varAlphaMixMeanImage, "var_alphamixmean");
				setupImage(_varSqrtBetaMixMeanImage, "var_sqrtbetamixmean");
				setupImage(_varDoseMeanImage, "var_dosemean");
				setupImage(_covAlphaMixMeanSqrtBetaMixMeanImage, "cov_alphamixmean_sqrtbetamixmean");
				setupImage(_covAlphaMixMeanDoseMeanImage, "cov_alphamixmean_dosemean");
				setupImage(_covSqrtBetaMixMeanDoseMeanImage, "cov_sqrtbetamixmean_dosemean");
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

		_voxelIndices.insert(index);
		_hitEventCountImage.AddValue(index, 1);

		if(_enableEdep) _edepImage.AddValue(index, eventEdep);

		_doseImage.AddValue(index, eventDose);
		_alphaMixImage.AddValue(index, eventAlphaMix);
		_sqrtBetaMixImage.AddValue(index, eventSqrtBetaMix);

		if(_enableUncertainty) {
			_squaredDoseImage.AddValue(index, eventDose * eventDose);
			_squaredAlphaMixImage.AddValue(index, eventAlphaMix * eventAlphaMix);
			_squaredSqrtBetaMixImage.AddValue(index, eventSqrtBetaMix * eventSqrtBetaMix);

			_alphaMixSqrtBetaMixImage.AddValue(index, eventAlphaMix * eventSqrtBetaMix);
			_alphaMixDoseImage.AddValue(index, eventAlphaMix * eventDose);
			_sqrtBetaMixDoseImage.AddValue(index, eventSqrtBetaMix * eventDose);
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

		if(alpha < 0) alpha = 0;
		if(sqrtBeta < 0) sqrtBeta = 0;

		// Accumulate alpha/beta
		_eventAlphaImage.AddValue(index, alpha);
		_eventSqrtBetaImage.AddValue(index, sqrtBeta);

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

		auto const alphaMixMean = _alphaMixImage.GetValue(index) / hitEventCount;
		auto const sqrtBetaMixMean = _sqrtBetaMixImage.GetValue(index) / hitEventCount;
		auto const dose = _doseImage.GetValue(index);
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

		if(_enableUncertainty) {
			if(scaledDose > 0 && alphaMixMean != 0 && sqrtBetaMixMean != 0 && sqrtDelta > 0 && _currentEvent > 0) {
				auto const doseMean = dose / n;
				auto const scaledDoseMean = scaledDose / n;
				auto const sqScaledDoseMean = sqScaledDose / n / n;
				auto const biodoseMean = biodose / n;
				auto const deltaMean = sqAlphaRef + 4 * _betaRef *
					(alphaMixMean * scaledDoseMean + sqrtBetaMixMean * sqrtBetaMixMean * sqScaledDoseMean);
				double sqrtDeltaMean = 0.;
				if(deltaMean >= 0)
					sqrtDeltaMean = std::sqrt(deltaMean);

				auto sumSquaredAlphaMix = _squaredAlphaMixImage.GetValue(index);
				auto sumSquaredSqrtBetaMix = _squaredSqrtBetaMixImage.GetValue(index);
				auto sumSquaredDose = _squaredDoseImage.GetValue(index);

				auto pdBiodoseAlphaMixMean = scaledDoseMean / sqrtDeltaMean;
				auto pdBiodoseSqrtBetaMixMean = 2 * sqScaledDoseMean * sqrtBetaMixMean / sqrtDeltaMean;
				auto pdBiodoseDoseMean = (
					(alphaMixMean * _doseScaleFactor) +
					2 * sqrtBetaMixMean * sqrtBetaMixMean * _doseScaleFactor * scaledDoseMean
				) / sqrtDeltaMean;

				auto varAlphaMixMean = (sumSquaredAlphaMix / hitEventCount - alphaMixMean * alphaMixMean) / hitEventCount;
				auto varSqrtBetaMixMean = (sumSquaredSqrtBetaMix / hitEventCount - sqrtBetaMixMean * sqrtBetaMixMean) / hitEventCount;
				auto varDoseMean = (sumSquaredDose / n - doseMean * doseMean) / n;

				auto sumAlphaMixSqrtBetaMix = _alphaMixSqrtBetaMixImage.GetValue(index);
				auto sumAlphaMixDose = _alphaMixDoseImage.GetValue(index);
				auto sumSqrtBetaMixDose = _sqrtBetaMixDoseImage.GetValue(index);
				auto covAlphaMixMeanSqrtBetaMixMean = (sumAlphaMixSqrtBetaMix / hitEventCount - alphaMixMean * sqrtBetaMixMean) / hitEventCount;
				auto covAlphaMixMeanDoseMean = (sumAlphaMixDose / n - alphaMixMean * doseMean) / n;
				auto covSqrtBetaMixMeanDoseMean = (sumSqrtBetaMixDose / n - sqrtBetaMixMean * doseMean) / n;

				auto partAlphaMix = pdBiodoseAlphaMixMean * pdBiodoseAlphaMixMean * varAlphaMixMean;
				auto partSqrtBetaMix = pdBiodoseSqrtBetaMixMean * pdBiodoseSqrtBetaMixMean * varSqrtBetaMixMean;
				auto partDose = pdBiodoseDoseMean * pdBiodoseDoseMean * varDoseMean;
				auto partAlphaMixSqrtBetaMix = 2 * pdBiodoseAlphaMixMean * pdBiodoseSqrtBetaMixMean * covAlphaMixMeanSqrtBetaMixMean;
				auto partAlphaMixDose = 2 * pdBiodoseAlphaMixMean * pdBiodoseDoseMean * covAlphaMixMeanDoseMean;
				auto partSqrtBetaMixDose = 2 * pdBiodoseSqrtBetaMixMean * pdBiodoseDoseMean * covSqrtBetaMixMeanDoseMean;

				auto uncertaintyDose = std::sqrt(varDoseMean) / doseMean;
				auto uncertaintyBiodose = std::sqrt(
					partAlphaMix + partSqrtBetaMix + partDose +
					partAlphaMixSqrtBetaMix + partAlphaMixDose + partSqrtBetaMixDose
				) / biodoseMean;

				if(_enableUncertaintyDetails) {
					_pdBiodoseAlphaMixMeanImage.SetValue(index, pdBiodoseAlphaMixMean);
					_pdBiodoseSqrtBetaMixMeanImage.SetValue(index, pdBiodoseSqrtBetaMixMean);
					_pdBiodoseDoseMeanImage.SetValue(index, pdBiodoseDoseMean);
					_varAlphaMixMeanImage.SetValue(index, varAlphaMixMean);
					_varSqrtBetaMixMeanImage.SetValue(index, varSqrtBetaMixMean);
					_varDoseMeanImage.SetValue(index, varDoseMean);
					_covAlphaMixMeanSqrtBetaMixMeanImage.SetValue(index, covAlphaMixMeanSqrtBetaMixMean);
					_covAlphaMixMeanDoseMeanImage.SetValue(index, covAlphaMixMeanDoseMean);
					_covSqrtBetaMixMeanDoseMeanImage.SetValue(index, covSqrtBetaMixMeanDoseMean);
				}

				_doseUncertaintyImage.SetValue(index, uncertaintyDose);
				_biodoseUncertaintyImage.SetValue(index, uncertaintyBiodose);
			} else {
				_doseUncertaintyImage.SetValue(index, 1);
				_biodoseUncertaintyImage.SetValue(index, 1);
			}
		}

		// Write data
		if(_enableDose)         _scaledDoseImage.SetValue(index, scaledDose);
		_bioDoseImage.SetValue(index, biodose);
		if(_enableRBE)          _rbeImage.SetValue(index, rbe);
	}
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::SaveData() {
	GateDebugMessageInc("Actor", 4, "GateBioDoseActor::SaveData() known ion events / total events: " << _eventWithKnownIonCount << " / " << _eventCount << "\n");

	updateData();

	GateVActor::SaveData();

	if(_enableEdep)           _edepImage.SaveData(_currentEvent);
	if(_enableDose)           _scaledDoseImage.SaveData(_currentEvent);
	if(_enableAlphaMix)       _alphaMixImage.SaveData(_currentEvent);
	if(_enableSqrtBetaMix)    _sqrtBetaMixImage.SaveData(_currentEvent);
	_bioDoseImage.SaveData(_currentEvent);
	if(_enableRBE)            _rbeImage.SaveData(_currentEvent);
	if(_enableUncertainty) {
		_doseUncertaintyImage.SaveData(_currentEvent);
	  _biodoseUncertaintyImage.SaveData(_currentEvent);

		if(_enableUncertaintyDetails) {
			_pdBiodoseAlphaMixMeanImage.SaveData(_currentEvent);
			_pdBiodoseSqrtBetaMixMeanImage.SaveData(_currentEvent);
			_pdBiodoseDoseMeanImage.SaveData(_currentEvent);
			_varAlphaMixMeanImage.SaveData(_currentEvent);
			_varSqrtBetaMixMeanImage.SaveData(_currentEvent);
			_varDoseMeanImage.SaveData(_currentEvent);
			_covAlphaMixMeanSqrtBetaMixMeanImage.SaveData(_currentEvent);
			_covAlphaMixMeanDoseMeanImage.SaveData(_currentEvent);
			_covSqrtBetaMixMeanDoseMeanImage.SaveData(_currentEvent);
		}
	}
	if(_enableHitEventCount)  _hitEventCountImage.SaveData(_currentEvent);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActor::ResetData() {
	_hitEventCountImage.Reset();

	_eventEdepImage.Reset();
	_eventDoseImage.Reset();
	_eventAlphaImage.Reset();
	_eventSqrtBetaImage.Reset();

	if(_enableEdep)         _edepImage.Reset();
	_doseImage.Reset();
	if(_enableDose) _scaledDoseImage.Reset();
	_alphaMixImage.Reset();
	_sqrtBetaMixImage.Reset();
	_bioDoseImage.Reset();
	if(_enableRBE)          _rbeImage.Reset();

	if(_enableUncertainty) {
		_doseUncertaintyImage.Reset();
		_biodoseUncertaintyImage.Reset();
		_squaredDoseImage.Reset();
		_squaredAlphaMixImage.Reset();
		_squaredSqrtBetaMixImage.Reset();
		_alphaMixSqrtBetaMixImage.Reset();
		_alphaMixDoseImage.Reset();
		_sqrtBetaMixDoseImage.Reset();

		if(_enableUncertaintyDetails) {
			_pdBiodoseAlphaMixMeanImage.Reset();
			_pdBiodoseSqrtBetaMixMeanImage.Reset();
			_pdBiodoseDoseMeanImage.Reset();
			_varAlphaMixMeanImage.Reset();
			_varSqrtBetaMixMeanImage.Reset();
			_varDoseMeanImage.Reset();
			_covAlphaMixMeanSqrtBetaMixMeanImage.Reset();
			_covAlphaMixMeanDoseMeanImage.Reset();
			_covSqrtBetaMixMeanDoseMeanImage.Reset();
		}
	}
}
//-----------------------------------------------------------------------------
