/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4ProductionCutsTable.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "GateMaterialTableToProductionCutsTable.hh"


GateMaterialTableToProductionCutsTable::GateMaterialTableToProductionCutsTable()
{
	m_nUpdateCounter=-1;
	Update();
}


GateMaterialTableToProductionCutsTable::~GateMaterialTableToProductionCutsTable()
{
}


void GateMaterialTableToProductionCutsTable::Update()
{
	const G4ProductionCutsTable* table=G4ProductionCutsTable::GetProductionCutsTable ();
	size_t	nMaterials = table->GetTableSize ();
	size_t nTotalMaterials=G4Material::GetNumberOfMaterials();

	m_oM2PVec.clear();
	m_oM2PVec.resize ( nTotalMaterials,-1 );

	m_oP2MVec.clear();
	m_oP2MVec.resize ( nMaterials,-1 );

	for ( size_t m=0; m<nMaterials; m++ )
	{
		const G4MaterialCutsCouple* couple=table->GetMaterialCutsCouple ( m );
		const G4Material* mat= couple->GetMaterial();
		assert(static_cast<G4int>(m)==couple->GetIndex());
		m_oM2PVec[mat->GetIndex() ]=m;
		m_oP2MVec[m]=mat->GetIndex();
	}
	m_nUpdateCounter++;
}

G4int GateMaterialTableToProductionCutsTable::M2P ( G4int i ) const
{
	if ( ( i<0 ) || ( i>=static_cast<G4int>(m_oM2PVec.size()) ) )
	{
		G4cout << "GateMaterialTableToProductionCutsTable::M2P(G4int i): Index out of range! Aborting." << G4endl;
		exit ( EXIT_FAILURE );
	}
	return M2P_nocheck ( i );
}
G4int GateMaterialTableToProductionCutsTable::P2M ( G4int i ) const
{
	if ( ( i<0 ) || ( i>=static_cast<G4int>(m_oP2MVec.size() )) )
	{
		G4cout << "GateMaterialTableToProductionCutsTable::P2M(G4int i): Index out of range! Aborting." << G4endl;
		exit ( EXIT_FAILURE );
	}
	return P2M_nocheck ( i );
}

