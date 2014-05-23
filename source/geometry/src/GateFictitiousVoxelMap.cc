/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4Material.hh"
#include "G4Region.hh"
#include "GateCrossSectionsTable.hh"
#include "GateFictitiousVoxelMap.hh"
#include "G4Material.hh"

using namespace std;

GateFictitiousVoxelMap::GateFictitiousVoxelMap ( G4Envelope* env )
		: GateVFictitiousMap ( env ),pGeometryVoxelReader ( NULL ),m_nHalfContainerDim ( -1.,-1.,-1. ),m_nVoxelDim ( -1.,-1.,-1. )
{
	m_nDeleteGeometryVoxelReader=false;
	m_nDeleteCrossSectionTable=false;
	m_nNx=-1;
	m_nNy=-1;
	m_nNz=-1;

}


GateFictitiousVoxelMap::~GateFictitiousVoxelMap()
{
	if ( m_nDeleteGeometryVoxelReader ) delete pGeometryVoxelReader;
}



void GateFictitiousVoxelMap::RegisterGeometryVoxelReader ( GateVGeometryVoxelReader* r, bool del )
{
	if ( (pGeometryVoxelReader!=NULL) && del )
	{
		G4Exception ( "GateFictitiousVoxelMap::RegisterGeometryVoxelReader ( GateVGeometryVoxelReader*, bool)", "GateGeometryVoxelReader already registered", FatalException,
		              "pGeometryVoxelReader not NULL!" );

	}
	pGeometryVoxelReader=r;
	m_nVoxelDim=r->GetVoxelSize();
	m_nHalfContainerDim[0]=m_nVoxelDim[0]*r->GetVoxelNx() /2.;
	m_nHalfContainerDim[1]=m_nVoxelDim[1]*r->GetVoxelNy() /2.;
	m_nHalfContainerDim[2]=m_nVoxelDim[2]*r->GetVoxelNz() /2.;
	m_nNx=r->GetVoxelNx();
	m_nNy=r->GetVoxelNy();
	m_nNz=r->GetVoxelNz();

	m_nDeleteGeometryVoxelReader=del;
}






void GateFictitiousVoxelMap::GetMaterials ( std::vector<G4Material*>& vec ) const
{
	G4int nx=pGeometryVoxelReader->GetVoxelNx();
	G4int ny=pGeometryVoxelReader->GetVoxelNy();
	G4int nz=pGeometryVoxelReader->GetVoxelNz();
	G4int size=nx*ny*nz;

	vec.clear();
	for ( G4int i=0;i<size;i++ )
	{
		G4Material* mat=pGeometryVoxelReader->GetVoxelMaterial ( i );
		bool found=false;
		for ( size_t l=0;l<vec.size();l++ )
		{
			if ( mat->GetIndex() ==vec[l]->GetIndex() )
			{
				found =true;
				break;

			}
		}
		if ( !found )
		{
			vec.push_back ( mat );
		}
	}
}

void GateFictitiousVoxelMap::Check() const
{
	if ( pGeometryVoxelReader ==NULL ) 		G4Exception ( "GateFictitiousVoxelMap::Check()", "GeometryVoxelReader not registered", FatalException,
		        "pGeometryVoxelReader==NULL!" );
	if ( m_nNx<=0) G4Exception ( "GateFictitiousVoxelMap::Check()", "Number of voxels too small", FatalException,
		        "m_nNx<=0 !" );
	if ( m_nNy<=0) G4Exception ( "GateFictitiousVoxelMap::Check()", "Number of voxels too small", FatalException,
		        "m_nNy<=0 !" );
	if ( m_nNz<=0) G4Exception ( "GateFictitiousVoxelMap::Check()", "Number of voxels too small", FatalException,
		        "m_nNz<=0 !" );
}
