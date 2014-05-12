/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateMaterialTableToProductionCutsTable_hh
#define GateMaterialTableToProductionCutsTable_hh 1

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

#include <vector>
#include "G4ios.hh"

class GateMaterialTableToProductionCutsTable
{
	public:
		GateMaterialTableToProductionCutsTable();
		~GateMaterialTableToProductionCutsTable();

		void Update();
		inline G4int GetUpdateCounter() const;
		inline G4int M2P_nocheck ( G4int i ) const;
		inline G4int P2M_nocheck ( G4int i ) const;
		G4int M2P ( G4int i ) const;
		G4int P2M ( G4int i ) const;

	private:
		std::vector<G4int> m_oM2PVec, m_oP2MVec;
		G4int m_nUpdateCounter;
};

inline G4int GateMaterialTableToProductionCutsTable::M2P_nocheck ( G4int i ) const
{
	return m_oM2PVec[i];
}
inline G4int GateMaterialTableToProductionCutsTable::P2M_nocheck ( G4int i ) const
{
	return m_oP2MVec[i];
}
inline G4int GateMaterialTableToProductionCutsTable::GetUpdateCounter() const
{
	return m_nUpdateCounter;
}

#endif
