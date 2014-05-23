/*----------------------
   OpenGATE Collaboration

   Didier Benoit <benoit@cppm.in2p3.fr>
   Franca Cassol Brunner <cassol@cppm.in2p3.fr>

   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

#include <vector>
#include <fstream>

#include "GateImageCT.hh"

GateImageCT::GateImageCT()
{
	m_multiplicityModule = 1;
	m_multiplicityPixel_1 = 1;
	m_multiplicityPixel_2 = 1;
	m_multiplicityPixel_3 = 1;
	m_data = 0;
	m_byte = 0;
	m_currentFrameID = 0;
}

GateImageCT::~GateImageCT()
{
	;
}

void GateImageCT::Reset( std::vector<size_t>& moduleByAxis,
	std::vector<size_t>& pixelByAxis )
{
	//Clean-up the result of a previous acquisition (if any)
	if( m_data )
	{
		delete m_data;
		m_data = 0;
	}

	//the number of pixel in the detector
	for( G4int i = 0; i != 3 ; ++i )
		m_multiplicityModule *= moduleByAxis[ i ];

	for( G4int i = 0; i != 3 ; ++i )
	{
		m_multiplicityPixel_1 *= pixelByAxis[ i ];
		m_multiplicityPixel_2 *= pixelByAxis[ i + 3 ];
		m_multiplicityPixel_3 *= pixelByAxis[ i + 6 ];
	}

	m_numberOfPixelByModule = ( m_multiplicityPixel_1 + m_multiplicityPixel_2
		+ m_multiplicityPixel_3 );
	m_numberOfPixel = m_numberOfPixelByModule * m_multiplicityModule;

	m_byte = m_numberOfPixel * sizeof( float );

	G4cout << G4endl;
	G4cout << "****" << G4endl;
	G4cout << "Number of Pixels in your detector : " << m_numberOfPixel
		   << G4endl;
	G4cout << "Number of bytes by projection : "
		   << m_byte / 1024.0
		   << " Kb " << G4endl;
	G4cout << "****" << G4endl;
	G4cout << G4endl;

	m_data = new float[ m_numberOfPixel ];

	if( !m_data )
	{
		G4cout << "you are in GateImageCT::Reset()" << G4endl;
		G4Exception( "GateImageCT::Reset", "reset", FatalException, "Could not allocate a new image (out of memory?)" );
	}
}

void GateImageCT::ClearData( G4int frameID )
{
	//store the image number
	m_currentFrameID = frameID;

	// Clear the data sets
	memset( m_data, 0, m_byte );
}

void GateImageCT::StreamOut( std::ofstream& outputDataFile )
{
	outputDataFile.write( reinterpret_cast<char*>( m_data ), m_byte );
}

void GateImageCT::Fill( size_t pixelID )
{
	m_data[ pixelID ]++;
}
