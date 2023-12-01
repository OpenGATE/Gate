#ifndef GATE_SOURCE_DIGITS_HITS_INCLUDE_GATEBIODOSEACTOR_HH
#define GATE_SOURCE_DIGITS_HITS_INCLUDE_GATEBIODOSEACTOR_HH

/*----------------------
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
	\class GateBioDoseActor
	\author Éloïse Salles, Alexis Pereda <alexis.pereda@clermont.in2p3.fr>, Yasmine Ali
*/

#include <G4NistManager.hh>

#include "GateBioDoseActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateVImageActor.hh"

class GateBioDoseActor: public GateVImageActor
{
public:
	struct Deposited {
		double alpha;
		double sqrtBeta;
		double energy;
		double dose;
	};

	struct Coefficients {
		double a, b;
	};

	struct AlphaBetaCoefficients {
		Coefficients alpha;
		Coefficients sqrtBeta;
	};

	using VoxelIndex = int;
	using DepositedMap = std::map<VoxelIndex, Deposited>;
	using VoxelIndices = std::set<VoxelIndex>;

	using Fragment = std::pair<int, double>;
	using AlphaBetaInterpolTable = std::map<Fragment, AlphaBetaCoefficients>;

	using EnergyMaxForZ = std::map<int, double>;

public:
	FCT_FOR_AUTO_CREATOR_ACTOR(GateBioDoseActor)

	//-----------------------------------------------------------------------------
	// Constructs the sensor
	void Construct() override;

	// G4
	void BeginOfRunAction(const G4Run* r) override;
	void EndOfRunAction(const G4Run* r) override;
	void BeginOfEventAction(const G4Event* event) override;
	void EndOfEventAction(const G4Event* event) override;
	void UserSteppingActionInVoxel(const int index, const G4Step* step) override;

	// Do nothing but needed because pure virtual
	void UserPreTrackActionInVoxel(const int, const G4Track*) override {}
	void UserPostTrackActionInVoxel(const int, const G4Track*) override {}

	//  Saves the data collected to the file
	void SaveData() override;
	void ResetData() override;

	// Scorer related
	void Initialize(G4HCofThisEvent*) override {}
	void EndOfEvent(G4HCofThisEvent*) override {}

	// Messenger
	void SetDoseScaleFactor(G4double doseScaleFactor) { _doseScaleFactor = doseScaleFactor; }
	void SetAlphaRef(G4double alphaRef) { _alphaRef = alphaRef; }
	void SetBetaRef(G4double betaRef) { _betaRef = betaRef; }
	void SetCellLine(G4String s) { _cellLine = std::move(s); }
	void SetBioPhysicalModel(G4String s) { _bioPhysicalModel = std::move(s); }
	void SetSOBPWeight(G4double d) { _sobpWeight = d; }

	void SetEnableEdep(bool e) { _enableEdep = e; }
	void SetEnableDose(bool e) { _enableDose = e; }
	void SetEnableBioDose(bool e) { _enableBioDose = e; }
	void SetEnableAlphaMix(bool e) { _enableAlphaMix = e; }
	void SetEnableSqrtBetaMix(bool e) { _enableSqrtBetaMix = e; }
	void SetEnableRBE(bool e) { _enableRBE = e; }
	void SetEnableUncertainties(bool e) { _enableUncertainties = e; }

	// Input database
	void BuildDatabase();
	static Coefficients Interpol(double x1, double x2, double y1, double y2);

protected:
	GateBioDoseActor(G4String name, G4int depth = 0);

private:
	//Counters
	int _currentEvent;

	GateBioDoseActorMessenger _messenger;

	EnergyMaxForZ _energyMaxForZ;

	// Initialisation
	G4String _dataBase;
	G4String _cellLine;
	G4String _bioPhysicalModel;
	double _alphaRef, _betaRef;
	double _doseScaleFactor = 1.;

	G4double _sobpWeight;

	// Maps
	DepositedMap _depositedMap;
	AlphaBetaInterpolTable _alphaBetaInterpolTable;

	VoxelIndices _eventVoxelIndices;

	// Images
	GateImageWithStatistic _eventCountImage;

	GateImageWithStatistic _bioDoseImage;
	GateImageWithStatistic _edepImage;
	GateImageWithStatistic _doseImage;
	GateImageWithStatistic _alphaMixImage;
	GateImageWithStatistic _sqrtBetaMixImage;
	GateImageWithStatistic _rbeImage;

	GateImageWithStatistic _biodoseUncertaintyImage;
	GateImageWithStatistic _eventEdepImage;
	GateImageWithStatistic _eventDoseImage;
	GateImageWithStatistic _squaredDoseImage;
	GateImageWithStatistic _eventAlphaImage;
	GateImageWithStatistic _squaredAlphaMixImage;
	GateImageWithStatistic _eventSqrtBetaImage;
	GateImageWithStatistic _squaredSqrtBetaMixImage;

	GateImageWithStatistic _alphaMixSqrtBetaMixImage;
	GateImageWithStatistic _alphaMixDoseImage;
	GateImageWithStatistic _sqrtBetaMixDoseImage;

	// Outputs
	bool _enableEdep;
	bool _enableDose;
	bool _enableBioDose;
	bool _enableAlphaMix;
	bool _enableSqrtBetaMix;
	bool _enableRBE;
	bool _enableUncertainties;

	// Extra information
	int _stepCount = 0;
	int _stepWithKnownIonCount = 0;
};

MAKE_AUTO_CREATOR_ACTOR(BioDoseActor, GateBioDoseActor)

#endif
