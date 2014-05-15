/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateVFictitiousMap.hh"
#include "G4Material.hh"
#include "G4Region.hh"
#include "G4VSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4FastTrack.hh"
#include "G4ios.hh"
#include "G4Box.hh"
#include <vector>
#include "GateCrossSectionsTable.hh"

GateVFictitiousMap::GateVFictitiousMap ( G4Envelope* env )
{
	pCrossSectionsTable=NULL;
	m_nDeleteCrossSectionTable=false;
//	pMaxMaterial=NULL;
	pEnvelope=env;
	if ( pEnvelope->GetNumberOfRootVolumes() !=1 )
	{
		G4Exception ( "GateVFictitiousMap::GateVFictitiousMap(const G4Material*, const G4Envelope*)", "InvalidSetup", FatalException,
		              "GateVFictitiousMap only for single volumed G4Envelopes/G4Regions!" );
	}
	std::vector<G4LogicalVolume*>::iterator it=pEnvelope->GetRootLogicalVolumeIterator();
	pLogicalVolume=*it;
	pSolid=pLogicalVolume->GetSolid();

#ifdef G4VERBOSE
	// Check if it's a Box
	const G4Box* check = dynamic_cast<const G4Box*> ( pSolid );
	if ( !check )
	{

		G4cout << "GateVFictitiousMap used for solid that is not a G4Box." << G4endl;
		pBox=NULL;
	}
	else
		pBox=check;
#endif 

}


GateVFictitiousMap::~GateVFictitiousMap()
{
 	if (m_nDeleteCrossSectionTable) delete pCrossSectionsTable;
}

/* obsolete
void GateVFictitiousMap::RegisterMaxMaterial ( G4Material* m )
{
	if ( pMaxMaterial!=NULL )
	{
		G4cout << "WARNING: GateFictitiousVoxelMap::RegisterMaxMaterial ( G4Material*): Material already set!" << G4endl;

	}
	pMaxMaterial=m;
}
*/


void GateVFictitiousMap::RegisterCrossSectionsTable ( const GateCrossSectionsTable* p, bool del )
{
	if ( ( pCrossSectionsTable!=NULL ) && del )
	{
		G4Exception ( "GateFictitiousVoxelMap::RegisterCrossSectionsTable ( GateCrossSectionsTable*, bool)", "CrossSectionsTable already registered", FatalException,
		              "pCrossSectionsTable not NULL!" );

	}
	pCrossSectionsTable=p;
	m_nDeleteCrossSectionTable=del;
}

