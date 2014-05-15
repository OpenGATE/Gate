/*----------------------
   OpenGATE Collaboration

   Didier Benoit <benoit@cppm.in2p3.fr>
   Franca Cassol Brunner <cassol@cppm.in2p3.fr>

   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \file GateImageCT.hh

  \brief Class GateImageCT
  \author Didier Benoit <benoit@cppm.in2p3.fr>
  \author Franca Cassol Brunner <cassol@cppm.in2p3.fr>
*/

#ifndef GATEIMAGECT_HH
#define GATEIMAGECT_HH

#include "globals.hh"
#include <vector>
#include "GateConfiguration.h"

class GateImageCT
{
	public:
		//! Constructor
		GateImageCT();
		//! Destructor
		~GateImageCT();

		//! Reset the image and prepare a new acquisition
		void Reset( std::vector<size_t>&, std::vector<size_t>& );

		//! Clear the matrix and prepare a new run
		void ClearData( G4int );

		//! Store a digi into a projection
		void Fill( size_t );

		void StreamOut( std::ofstream& );

		inline G4int GetBytesByImage()
		{ return m_byte; }

		//! Setters and Getters
		inline size_t GetNumberOfPixel()
		{ return m_numberOfPixel; }

		inline size_t GetNumberOfPixelByModule()
		{ return m_numberOfPixelByModule; }

		inline size_t GetMultiplicityModule()
		{ return m_multiplicityModule; }

		inline size_t GetMultiplicityPixel1()
		{ return m_multiplicityPixel_1; }

		inline size_t GetMultiplicityPixel2()
		{ return m_multiplicityPixel_2; }

		inline size_t GetMultiplicityPixel3()
		{ return m_multiplicityPixel_3; }

		inline G4int GetCurrentFrame()
		{ return m_currentFrameID; }

	private:
		float* m_data;
		size_t m_multiplicityModule;
		size_t m_numberOfPixel;
		size_t m_numberOfPixelByModule;
		size_t m_multiplicityPixel_1;
		size_t m_multiplicityPixel_2;
		size_t m_multiplicityPixel_3;
		G4int m_byte;
		G4int m_currentFrameID;
};

#endif
