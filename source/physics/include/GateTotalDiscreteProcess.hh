/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

#ifndef GateTotalDiscreteProcess_hh
#define GateTotalDiscreteProcess_hh

#include "G4VDiscreteProcess.hh"
#include "G4VEmProcess.hh"
#include <vector>
#include "G4ios.hh"
class G4VParticleChange;
class G4Track;
#include "G4ForceCondition.hh"
class G4ParticleDefinition;
class G4Step;
class G4String;
#include "G4Gamma.hh"
#include "G4PhysicsTable.hh"
#include "GateCrossSectionsTable.hh"
#include "CLHEP/Random/RandFlat.h" 
#include <iostream>
#include "Randomize.hh"
#include <cassert>

class GateTotalDiscreteProcess : public G4VDiscreteProcess
{
	public:
		GateTotalDiscreteProcess ( const G4String& name, G4ProcessType type, G4int num_processes, const G4ParticleDefinition* particle_type, G4double minEnergy, G4double maxEnergy, G4int binNumbers );
		virtual ~GateTotalDiscreteProcess();
		bool AddDiscreteProcess (G4VDiscreteProcess* p );


		G4VParticleChange * 	PostStepDoIt ( const G4Track &track, const G4Step &stepData );


		inline G4double 	PostStepGetPhysicalInteractionLength ( const G4Track &track, G4double previousStepSize, G4ForceCondition *condition );

		void 	BuildPhysicsTable ( const G4ParticleDefinition & );
		inline G4double GetNumberOfInteractionLengthLeft() const;
		inline G4bool 	IsApplicable ( const G4ParticleDefinition & );
		const GateCrossSectionsTable* GetTotalCrossSectionsTable() const;
		G4VProcess* GetActiveProcess() const;
		G4VProcess* SampleActiveProcess(G4Material*,G4double energy);

	protected:
		G4double GetMeanFreePath ( const G4Track &aTrack, G4double previousStepSize, G4ForceCondition *condition );

		void CreateTotalMaxCrossSectionTable(const std::vector<G4Material*>&);

		void BuildCrossSectionsTables();

		G4int m_nNumProcesses, m_nMaxNumProcesses;
		bool m_nInitialized;
		const G4ParticleDefinition* pParticleType;
		std::vector<G4VDiscreteProcess*> m_oProcessVec;
		std::vector<G4String*> m_oProcessNameVec;
		std::vector<GateCrossSectionsTable*> m_oCrossSectionsTableVec;
		GateCrossSectionsTable* m_pTotalCrossSectionsTable;
		G4double m_nTotalMinEnergy, m_nTotalMaxEnergy;
		G4int m_nTotalBinNumber; // binning information
		size_t m_nProcessWithSmallestPIL;
};





inline G4bool GateTotalDiscreteProcess::IsApplicable ( const G4ParticleDefinition &p )
{
	return ( &p==pParticleType ); // singleton pointer comparison should be sufficient (faster!)
}

inline G4double GateTotalDiscreteProcess::GetNumberOfInteractionLengthLeft () const
{
	return theNumberOfInteractionLengthLeft;
}


inline G4double 	GateTotalDiscreteProcess::PostStepGetPhysicalInteractionLength ( const G4Track &track, G4double previousStepSize, G4ForceCondition *condition )
{
	G4double b=m_oProcessVec[0]->PostStepGetPhysicalInteractionLength ( track, previousStepSize, condition );
	m_nProcessWithSmallestPIL=0;
	for (size_t i=1;i<m_oProcessVec.size();i++)
	{
		
		const G4double a=m_oProcessVec[i]->PostStepGetPhysicalInteractionLength (track, previousStepSize, condition);
		if (a<b)
		{
			b=a;
			m_nProcessWithSmallestPIL=i;
		}
	}
	return b;
}

inline G4VProcess* GateTotalDiscreteProcess::GetActiveProcess() const
{
	return m_oProcessVec[m_nProcessWithSmallestPIL];
}

inline G4VProcess* GateTotalDiscreteProcess::SampleActiveProcess(G4Material* m,G4double energy)
{
	G4double tot=m_pTotalCrossSectionsTable->GetCrossSection(m,energy);
	size_t i=m_nNumProcesses;
	G4double rand=G4UniformRand()*tot;
	G4double p=m_oCrossSectionsTableVec[--i]->GetCrossSection(m,energy);
	while (rand>p)
	{ p+=m_oCrossSectionsTableVec[--i]->GetCrossSection(m,energy);};
	m_nProcessWithSmallestPIL=i;
	return m_oProcessVec[i];
}








#endif
