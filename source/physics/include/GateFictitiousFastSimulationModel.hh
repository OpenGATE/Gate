/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateFictitiousFastSimulationModel_hh
#define GateFictitiousFastSimulationModel_hh 1

#include <G4VFastSimulationModel.hh>

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

class GateFictitiousVoxelMap;
class GateCrossSectionsTable;
class GateTotalDiscreteProcess;
class G4Material;
#include "G4ios.hh"
class G4VSolid;
#include "GatePETVRTSettings.hh"
class G4StepPoint;
#include "G4ThreeVector.hh"
class GatePhantomSD;
#include "G4Track.hh"
class G4Step;
#include "G4TrackFastVector.hh"


class GateFictitiousFastSimulationModel : public G4VFastSimulationModel
{
	public:
	
		GateFictitiousFastSimulationModel ( G4double absoluteMinEnergy, G4double absoluteMaxEnergy );
		~GateFictitiousFastSimulationModel();

		bool SetParameters(const GatePETVRTSettings* settings); // returns true if everything initialized

		inline bool Initialized() const;

		void SetMinEnergy(G4double);
		void SetMaxEnergy(G4double);

		void AttachPhantomSD(GatePhantomSD*);
		G4bool IsApplicable ( const G4ParticleDefinition& );
		G4bool ModelTrigger ( const G4FastTrack & );
		void DoIt ( const G4FastTrack&, G4FastStep& );

	private:
		void SetTotalDiscreteProcess ( GateTotalDiscreteProcess* );
		void SetFictitiousMap ( GateVFictitiousMap* );
		inline void SetApproximations ( GatePETVRT::Approx );
		void StepwiseTrace();
		void VolumeTrace();
		inline void Affine ( G4ThreeVector& current, const G4ThreeVector& dir, const G4double length) const;
		//void SetCrossSectionsTable ( const GateCrossSectionsTable* );
		inline void AddSecondaries(G4VParticleChange* change);
		inline void AddSecondariesToFastStep();
		inline void DiscardPhoton();
		const GateCrossSectionsTable* pTotalCrossSectionsTable;
		GateVFictitiousMap* pFictitiousMap;
		GateTotalDiscreteProcess* pTotalDiscreteProcess;
		//const G4Material* pMaxMaterial;
		const G4VSolid* pEnvelopeSolid;
		G4double m_nMinEnergy, m_nMaxEnergy;
		G4double m_nDiscardEnergy;
		GatePETVRT::Approx m_nApproximations;
		bool m_nInitialized,m_Initialized2nd;
		const G4FastTrack* pCurrentFastTrack;
		G4Track* pCurrentTrack;
		G4Step* pCurrentTrackStep;
		G4FastStep* pCurrentFastStep;
		G4double m_nCurrentInvFictCrossSection;
		G4double m_nCurrentEnergy;
		G4double m_nPathLength;
		G4double m_nTotalPathLength;
		G4double m_nDistToOut;
		G4double m_nTime;
		G4double m_nCurrentVelocity;
		G4ThreeVector m_nCurrentLocalPosition;
		G4ThreeVector m_nCurrentLocalDirection;
		GatePhantomSD* pPhantomSD;
		G4TrackFastVector* m_pTrackFastVector;
		G4int m_nNumSecondaries;
		G4double m_nSurfaceTolerance;
		const G4AffineTransform* pInverseTransform;
		const G4AffineTransform* pTransform;
		G4double m_nAbsMinEnergy;
		G4double m_nAbsMaxEnergy;
}
;

inline bool GateFictitiousFastSimulationModel::Initialized() const
{
	return m_nInitialized;
}
inline void GateFictitiousFastSimulationModel::SetApproximations ( GatePETVRT::Approx b )
{
	m_nApproximations=b;
}
inline void GateFictitiousFastSimulationModel::Affine ( G4ThreeVector& current, const G4ThreeVector& dir, const G4double length) const
{
		current[0]+=dir[0]*length;
		current[1]+=dir[1]*length;
		current[2]+=dir[2]*length;
}

inline void GateFictitiousFastSimulationModel::AddSecondaries(G4VParticleChange* change)
{
	G4int numSec=change->GetNumberOfSecondaries();
	G4int numTotal=numSec+m_nNumSecondaries;
	if (numTotal>=G4TrackFastVectorSize)
				G4Exception ( "GateFictitiousFastSimulationModel::AddSecondaries(G4VParticleChange* change) const", "Too many secondaries (>512)!", FatalException,
			              "Not foreseen." );

	pCurrentFastStep->SetNumberOfSecondaryTracks ( numSec );
	for ( G4int i=0;i<numSec;i++ )
	{
		change->GetSecondary ( i )->SetGlobalTime ( m_nTime );
		m_pTrackFastVector->SetElement(i+m_nNumSecondaries, (change->GetSecondary ( i )));
	}
	m_nNumSecondaries=numTotal;
	change->Clear();
}

inline void GateFictitiousFastSimulationModel::AddSecondariesToFastStep()
{
		pCurrentFastStep->SetNumberOfSecondaries(m_nNumSecondaries);
		for ( G4int i=0;i<m_nNumSecondaries;i++ )
			pCurrentFastStep->AddSecondary((*m_pTrackFastVector)[i]);
}

inline void GateFictitiousFastSimulationModel::DiscardPhoton()
{
				pCurrentFastStep->KillPrimaryTrack();
				pCurrentFastStep->ProposePrimaryTrackPathLength(0.0);
  				pCurrentFastStep->ProposeTotalEnergyDeposited(m_nCurrentEnergy);
}
#endif

