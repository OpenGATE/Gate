/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"

#include "GateTessellated.hh"
#include "GateTessellatedMessenger.hh"

GateTessellated::GateTessellated( G4String const& itsName, G4String const& itsMaterialName )
  : GateVVolume( itsName, false, 0 ),
    m_tessellated_solid( NULL ),
    m_tessellated_log( NULL ),
    m_tessellated_phys( NULL ),
    m_PathToVerticesFile( "" ),
    m_Messenger( NULL )
{
  SetMaterialName( itsMaterialName );
  m_Messenger = new GateTessellatedMessenger( this );
}

GateTessellated::GateTessellated( G4String const& itsName, G4bool itsFlagAcceptChildren, G4int depth )
  : GateVVolume( itsName, itsFlagAcceptChildren, depth ),
    m_tessellated_solid( NULL ),
    m_tessellated_log( NULL ),
    m_tessellated_phys( NULL ),
    m_PathToVerticesFile( "" ),
    m_Messenger( NULL )
{
  SetMaterialName( "Vacuum" );
  m_Messenger = new GateTessellatedMessenger( this );
}

GateTessellated::~GateTessellated()
{
  delete m_Messenger;
}

G4LogicalVolume* GateTessellated::ConstructOwnSolidAndLogicalVolume( G4Material* mater, G4bool flagUpdateOnly )
{
  G4ThreeVector v1, v2, v3;
  if( !flagUpdateOnly || !m_tessellated_solid )
  {
    m_tessellated_solid = new G4TessellatedSolid( GetSolidName() );
    std::ifstream verticesFile( m_PathToVerticesFile, std::ios::in );
    if( !verticesFile )
    {
      G4cerr << "No vertices file: " << m_PathToVerticesFile << G4endl;
    }
    else
    {
      while( true )
      {
        verticesFile >> v1[ 0 ] >> v1[ 1 ] >> v1[ 2 ] >> v2[ 0 ] >> v2[ 1 ] >> v2[ 2 ] >> v3[ 0 ] >> v3[ 1 ] >> v3[ 2 ];
        if( verticesFile.eof() ) break;
        AddFacet( v1, v2, v3 );
      }
      m_tessellated_solid->SetSolidClosed( true );
      verticesFile.close();
      m_tessellated_log = new G4LogicalVolume( m_tessellated_solid, mater, GetLogicalVolumeName() );
    }
  }

  return m_tessellated_log;
}

void GateTessellated::DestroyOwnSolidAndLogicalVolume()
{
  if( m_tessellated_log ) delete m_tessellated_log;
  m_tessellated_log = NULL;

  if( m_tessellated_solid ) delete m_tessellated_solid;
  m_tessellated_solid = NULL;
}

void GateTessellated::SetPathToVerticesFile( G4String path )
{
  m_PathToVerticesFile = path;
}

void GateTessellated::AddFacet( G4ThreeVector vertice1, G4ThreeVector vertice2, G4ThreeVector vertice3 )
{
  G4TriangularFacet *facet = new G4TriangularFacet( vertice1, vertice2, vertice3, ABSOLUTE );
  m_tessellated_solid->AddFacet( (G4VFacet*)facet );
}
