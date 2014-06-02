/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateFictitiousVoxelMap_hh
#define GateFictitiousVoxelMap_hh 1

#include "GateVGeometryVoxelReader.hh"
#include "GateVFictitiousMap.hh"
#include <cassert>
#include "G4ios.hh"

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

class G4Material;
class G4Region;
#include "GateCrossSectionsTable.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include "G4GeometryTolerance.hh" 

typedef G4Region G4Envelope;

class GateFictitiousVoxelMap : public GateVFictitiousMap
{
	public:
		GateFictitiousVoxelMap ( G4Envelope* env );
		~GateFictitiousVoxelMap();
		void RegisterGeometryVoxelReader ( GateVGeometryVoxelReader *, bool deleteWithThis );
		inline const GateCrossSectionsTable* GetCrossSectionsTable() const;
		inline G4double GetCrossSection ( const G4ThreeVector& pos, G4double kin_en ) const;
		inline G4double GetMaxCrossSection ( G4double kin_en ) const;
		inline G4int GetNx() const;
		inline G4int GetNy() const;
		inline G4int GetNz() const;
		void Check() const;

		inline G4Material* GetMaterial ( const G4ThreeVector& pos ) const;
		void GetMaterials ( std::vector<G4Material*>& ) const;
		inline GateVGeometryVoxelReader* GetGeometryVoxelReader() const;

	private:
		GateVGeometryVoxelReader * pGeometryVoxelReader;
		G4ThreeVector m_nHalfContainerDim; // half dimensions of container
		G4ThreeVector m_nVoxelDim; // dimension of voxel
		bool m_nDeleteGeometryVoxelReader;
		G4int m_nNx, m_nNy, m_nNz;
};

inline G4int GateFictitiousVoxelMap::GetNx() const
{
	return m_nNx;
}
inline G4int GateFictitiousVoxelMap::GetNy() const
{
	return m_nNy;
}
inline G4int GateFictitiousVoxelMap::GetNz() const
{
	return m_nNz;
}


inline const GateCrossSectionsTable* GateFictitiousVoxelMap::GetCrossSectionsTable() const
{
	return pCrossSectionsTable; 
}
inline G4Material* GateFictitiousVoxelMap::GetMaterial ( const G4ThreeVector& localPos ) const
{
	G4int i=static_cast<G4int> ( ( localPos[0]+m_nHalfContainerDim[0] ) /m_nVoxelDim[0] );
	G4int j=static_cast<G4int> ( ( localPos[1]+m_nHalfContainerDim[1] ) /m_nVoxelDim[1] );
	G4int k=static_cast<G4int> ( ( localPos[2]+m_nHalfContainerDim[2] ) /m_nVoxelDim[2] );

	if ( i>=m_nNx ){
		if ( localPos[0]+m_nHalfContainerDim[0]-m_nVoxelDim[0]*m_nNx<=G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			i=m_nNx-1;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "x position outside fictitious volume", FatalException,
			              "x too large" );
        }
	if ( j>=m_nNy ){
		if ( localPos[1]+m_nHalfContainerDim[1]-m_nVoxelDim[1]*m_nNy<=G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			j=m_nNy-1;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "y position outside fictitious volume", FatalException,
			              "y too large" );
        }

	if ( k>=m_nNz ){
		if ( localPos[2]+m_nHalfContainerDim[2]-m_nVoxelDim[2]*m_nNz<=G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			k=m_nNz-1;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "z position outside fictitious volume", FatalException,
			              "z too large" );
        }
	if ( i<0 ){
		if ( localPos[0]+m_nHalfContainerDim[0]-m_nVoxelDim[0]*m_nNx>-G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			i=0;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "x position outside fictitious volume", FatalException,
			              "x too small" );
        }
	if ( j<0 ){
		if ( localPos[1]+m_nHalfContainerDim[1]-m_nVoxelDim[1]*m_nNy>-G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			j=0;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "y position outside fictitious volume", FatalException,
			              "y too small" );
        }
	if ( k<0 ){
		if ( localPos[2]+m_nHalfContainerDim[2]-m_nVoxelDim[2]*m_nNz>-G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() )
			k=0;
		else
			G4Exception ( "GateFictitiousVoxelMap::GetMaterial (const G4ThreeVector& localPos )", "z position outside fictitious volume", FatalException,
			              "z too small" );
        }

	return pGeometryVoxelReader->GetVoxelMaterial_noCheck ( i,j,k );
}

inline G4double GateFictitiousVoxelMap::GetCrossSection ( const G4ThreeVector& pos, G4double kin_en ) const
{
	if ( pCrossSectionsTable==NULL ) 		G4Exception ( "GateFictitiousVoxelMap::GetCrossSection ( const G4ThreeVector&, G4double)", "CrossSectionsTable not registered", FatalException,
		        "pCrossSectionsTable==NULL!" );

	return pCrossSectionsTable->GetCrossSection ( GetMaterial ( pos ),kin_en );
}

inline GateVGeometryVoxelReader* GateFictitiousVoxelMap::GetGeometryVoxelReader() const
{
	return pGeometryVoxelReader;
}

inline G4double GateFictitiousVoxelMap::GetMaxCrossSection ( G4double kin_en ) const
{
	return pCrossSectionsTable->GetMaxCrossSection ( kin_en );
}


#endif
