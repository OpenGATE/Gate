/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateCrossSectionsTable_hh
#define GateCrossSectionsTable_hh 1

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/


//#include "G4PhysicsTable.hh"

#include <vector>
#include "G4PhysicsTable.hh"
#include "G4Material.hh"
class G4ParticleDefinition;
class G4EmCalculator;
#include "G4Gamma.hh"
#include <iostream>
#include "GateMaterialTableToProductionCutsTable.hh"
#include <vector>
#include "G4VDiscreteProcess.hh"
class G4MaterialCutsCouple;

class GateCrossSectionsTable: public G4PhysicsTable
{
	public:
		GateCrossSectionsTable ( G4double minEnergy, G4double maxEnergy, G4int physicsVectorBinNumber, const G4ParticleDefinition* pdef, G4VDiscreteProcess& process );

		GateCrossSectionsTable ( G4double minEnergy, G4double maxEnergy, G4int physicsVectorBinNumber, const G4ParticleDefinition* pdef, const std::vector<G4VDiscreteProcess*>& processes );

		GateCrossSectionsTable ( std::ifstream& fin,bool ascii, const std::vector<G4VDiscreteProcess*>& processes );


		~GateCrossSectionsTable();

		size_t SetAndBuildProductionMaterialTable(); // returns index of last material
		bool CheckInternalProductionMaterialTable() const; // should return true if internal tables correctly initialized

		bool BuildMaxCrossSection ( const std::vector<G4Material*>& ); // turns this table into a fictitious table

		const G4Material* GetMaterial ( size_t index ) const;
		G4int GetIndex ( const G4Material* ) const;

		inline G4double GetMinEnergy() const;
		inline G4double GetMaxEnergy() const;
		inline int GetVerbose() const;
		void SetVerbose(int v);

		G4double GetEnergyLimitForGivenMaxCrossSection(G4double crossSection) const;
		void StoreTable ( std::ofstream& out, bool ascii ) const;
		void RetrieveTable ( std::ifstream& in, bool ascii );

		inline G4double GetCrossSection ( const G4Material*, G4double energy ) const;
		inline G4double GetCrossSection ( const G4Material*, G4double energy, G4double density ) const;
		inline G4double GetCrossSection ( size_t materialIndex, G4double energy ) const;
		inline G4double GetCrossSection ( size_t materialIndex, G4double energy, G4double density ) const;	//assuming that cross section linear in density (not yet implemented)
		inline G4double GetMaxCrossSection ( G4double energy ) const;

	protected:
		size_t AddMaterial ( const G4MaterialCutsCouple* ); // returns index for that material




		G4double m_nMinEnergy, m_nMaxEnergy; // of G4PhysicsVectors
		G4int m_nPhysicsVectorBinNumber;
		const G4Material* pReplacementMaterial;

		std::vector<G4double> m_oInvDensity;
		std::vector<const G4Material*> m_oMaterialVec;
		const std::vector<G4VDiscreteProcess*> m_oProcessVec;// several if cross section should be total
		G4PhysicsVector* m_pMaxCrossSection;
		int m_nVerbose;

		const G4ParticleDefinition* pParticleDefinition;
		static G4int PARTICLE_NAME_LENGTH; // for ParticleName and Retrieve and Store Table if writing/reading in binary

		void Store ( std::ofstream&, bool ascii, size_t num ) const;
		void Retrieve ( std::ifstream&, bool ascii, size_t num );

		static G4EmCalculator m_sEmCalculator;
		//const std::vector<const G4String*> m_oProcessNameVec; // several if cross section should be total

		GateMaterialTableToProductionCutsTable* pMaterialTableToProductionCutsTable;


};
inline G4double GateCrossSectionsTable::GetMinEnergy() const
{
	return m_nMinEnergy;
}

inline G4double GateCrossSectionsTable::GetMaxEnergy() const
{
	return m_nMaxEnergy;
}

inline int GateCrossSectionsTable::GetVerbose() const
{
	return m_nVerbose;
}

inline G4double GateCrossSectionsTable::GetMaxCrossSection ( G4double energy ) const
{
	bool NotUsedAnyMoreIsOutOfRange;
	return m_pMaxCrossSection->GetValue ( energy,NotUsedAnyMoreIsOutOfRange );
}

inline G4double GateCrossSectionsTable::GetCrossSection ( size_t materialIndex, G4double energy) const
{
	bool NotUsedAnyMoreIsOutOfRange;
	assert ( size() >materialIndex );
	assert ( energy>=m_nMinEnergy );
	assert ( energy<m_nMaxEnergy );
	//G4cout << "material no " << materialIndex << " energy "<< energy << G4endl;
	return operator() ( materialIndex )->GetValue ( energy,NotUsedAnyMoreIsOutOfRange );
}

inline G4double GateCrossSectionsTable::GetCrossSection ( size_t materialIndex, G4double energy, G4double density) const
{
	bool NotUsedAnyMoreIsOutOfRange;
	assert ( size() >materialIndex );
	assert ( energy>=m_nMinEnergy );
	assert ( energy<m_nMaxEnergy );
	return operator() ( materialIndex )->GetValue ( energy,NotUsedAnyMoreIsOutOfRange )*m_oInvDensity[materialIndex]*density;
       }

       inline G4double GateCrossSectionsTable::GetCrossSection ( const G4Material* mat, G4double energy) const
{
	//return GetCrossSection ( pMaterialTableToProductionCutsTable->M2P ( mat->GetIndex()),energy);
	return GetCrossSection ( pMaterialTableToProductionCutsTable->M2P ( mat->GetIndex()),energy);
}

inline G4double GateCrossSectionsTable::GetCrossSection ( const G4Material* mat, G4double energy,G4double density) const
{
	//return GetCrossSection ( pMaterialTableToProductionCutsTable->M2P ( mat->GetIndex()),energy,density);
	return GetCrossSection ( pMaterialTableToProductionCutsTable->M2P ( mat->GetIndex()),energy,density);
}

inline  const G4Material* GateCrossSectionsTable::GetMaterial ( size_t index ) const
{
	assert ( index<m_oMaterialVec.size() );
	return m_oMaterialVec[index];
}



#endif
