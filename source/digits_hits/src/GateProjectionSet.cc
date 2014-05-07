/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \file GateProjectionSet.hh

  $Log: GateProjectionSet.cc,v $
  Revision 1.1.1.1.4.1  2011/03/10 16:32:35  henri
  Implemented multiple energy window interfile output

  Revision 1.1.1.1.4.1  2011/02/02 15:37:46  henri
  Added support for multiple energy windows

  Revision 1.3  2010/12/01 17:11:23  henri
  Various bug fixes

  Revision 1.2  2010/11/26 14:26:18  henri
  Class GateProjectionSet

  Changes for adding multiple energy window support. The projection data is now a 3D array, storing each energy window and each head for 1 camera position.

  Added attribute :
  	- size_t m_energyWindowNb to store the total of energy windows

  Modified attributes :
  	- ProjectionDataType** m_data -> ProjectionDataType*** m_data (was [head][pixel], now is [energyWindow][head][pixel]
  	- ProjectionDataType* m_dataMax -> ProjectionDataType** m_dataMax (was [head], now is [energyWindow][head])

  Methods modified with no prototype change:
  	- ClearData(size_t) : Clearing of 3D array instead of 3D array

  Methods modified which prototype changed :
  	- Reset(size_t headNb, size_t projectionNb) -> Reset(size_t energyWindowNb, size_t headNb, size_t projectionNb) : Allocation of 3D array regarding to the new m_data and m_dataMax type
  	- Fill(G4int, G4double x, G4double y) -> Fill(G4int energyWindowID, G4int headID, G4double, G4double)
  	- StreamOut(std::ofstream&, size_t headID) -> StreamOut(std::ofstream&, size_t energyWindowID, size_t headID)



*/


#include "GateProjectionSet.hh"

#include "globals.hh"
#include "G4UnitsTable.hh"
#include "GateARFSD.hh"
#include "GateSPECTHeadSystem.hh"

// Reset the matrix and prepare a new acquisition
void GateProjectionSet::Reset(size_t energyWindowNumber, size_t headNumber,size_t projectionNumber)
{
   m_rec = 0;
   m_rej = 0;
   size_t energyWindowID;
   size_t headID;

  // Fist clean-up the result of a previous acqisition (if any)
  // Modified by HDS : we now need to clean each head data for each energy window
  if (m_data) {
    for (energyWindowID = 0; energyWindowID < m_energyWindowNb; energyWindowID++) {
    	for (headID=0;headID<m_headNb;headID++) {
      		free(m_data[energyWindowID][headID]);
    	}
    	free(m_data[energyWindowID]);
    }

    free(m_data);
    m_data=0;
  }

  // We also need to clean all max data for each energy window
	if (m_dataMax) {
  		for (energyWindowID = 0; energyWindowID < m_energyWindowNb; energyWindowID++) {
    		free(m_dataMax[energyWindowID]);
    	}
    	free(m_dataMax);
    }

  // Store the new number of projections
  m_energyWindowNb = energyWindowNumber;
  m_headNb = headNumber;
  m_projectionNb = projectionNumber;
  m_currentProjectionID = -1;

  if ( (!m_energyWindowNb) || (!m_headNb) || (!m_projectionNb) )
    return;

GateDetectorConstruction* theDC = GateDetectorConstruction::GetGateDetectorConstruction();

G4int stage = -2;

if ( theDC->GetARFSD() != 0  ) stage = theDC->GetARFSD()->GetStage();

if ( stage == 2 )
{

 	G4cout << " GateProjectionSet::Reset detected ARF tables for Production Use"<<G4endl;

 	G4cout << " m_ARFdata = " <<m_ARFdata<<G4endl;

  if (m_ARFdata) {
    for (headID=0;headID<m_headNb;headID++) {
      free(m_ARFdata[headID]);
    }
    free(m_ARFdata);
    m_ARFdata=0;
  }

    if (m_ARFdataMax)
    free(m_ARFdataMax);

    m_ARFdata = (ARFProjectionDataType**) malloc( m_headNb * sizeof(ARFProjectionDataType*) );
    if (!m_ARFdata) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a new projection set for ARF Projections (out of memory?)\n");

    for (headID=0;headID<m_headNb;headID++)
     {
      m_ARFdata[headID] = (ARFProjectionDataType*) malloc( ARFBytesPerProjection() );
      if (!m_ARFdata[headID]) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a new projection set (out of memory?)");
      G4cout << " ARF projection bins allocated at " << m_ARFdata[headID] << G4endl;
     }

  // Allocate the data-max pointer
  m_ARFdataMax = (ARFProjectionDataType*) calloc( m_headNb , ARFBytesPerPixel() );
  if (!m_ARFdataMax) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a statistics array (out of memory?)\n");

 G4cout << " GateProjectionSet::Reset : Estimated size for the Binary Projection Output file " <<ARFBytesPerHead() * G4double(m_headNb) / ( 1024.* 1024. )<<" Mo"<<G4endl;
G4cout << " GateProjectionSet::Reset : Estimated size for the Binary Projection Output file " <<ARFBytesPerHead() * G4double(m_headNb) / 1024.<<" Ko"<<G4endl;
 return;
}

  if (m_verboseLevel>2)
    G4cout << "Allocating " << m_headNb << " projection matrices " << m_pixelNbX << " x " << m_pixelNbY << G4endl;

  // Allocate the data pointer
  // Modified by HDS : allocation of a 3D array
  m_data = (ProjectionDataType***) malloc( m_energyWindowNb * sizeof(ProjectionDataType**)  );
  if (!m_data) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a new projection set (out of memory?)");

  // Do the allocations for each energy window
  	for (energyWindowID = 0; energyWindowID < m_energyWindowNb; energyWindowID++) {
		m_data[energyWindowID] = (ProjectionDataType**) malloc( m_headNb * sizeof(ProjectionDataType*) );
  		if (!m_data[energyWindowID]) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a new projection (out of memory?)");

 		 // Do the allocations for each head
		for (headID=0;headID<m_headNb;headID++) {
			m_data[energyWindowID][headID] = (ProjectionDataType*) malloc( BytesPerProjection() );
    		if (!(m_data[energyWindowID][headID])) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a new projection (out of memory?)");
      	}
  	}

  // Allocate the data-max pointer
  m_dataMax =  (ProjectionDataType**) malloc( m_energyWindowNb * sizeof(ProjectionDataType*) );
  if (!m_dataMax) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a statistics array (out of memory?)");

  for (energyWindowID = 0; energyWindowID < m_energyWindowNb; energyWindowID++) {
  	m_dataMax[energyWindowID] = (ProjectionDataType*) calloc( m_headNb , BytesPerPixel() );
 	if (!m_dataMax[energyWindowID]) G4Exception( "GateProjectionSet::Reset", "Reset", FatalException, "Could not allocate a statistics array (out of memory?)\n");
  }

}



// Clear the matrix and prepare a new run
// Modified by HDS : multiple energy window support
void GateProjectionSet::ClearData(size_t projectionID)
{
  // Store the projection ID
  m_currentProjectionID = projectionID;

  GateDetectorConstruction* theDC = GateDetectorConstruction::GetGateDetectorConstruction();

G4int arfstage = -3;


  // Clear the data sets
  if ( theDC->GetARFSD() != 0 ) arfstage = theDC->GetARFSD()->GetStage();

if ( arfstage  == 2 )
{
  for (size_t headID=0;headID<m_headNb;headID++)
  memset(m_ARFdata[headID],0, ARFBytesPerProjection() );
}
else
{ G4cout << " clearing up the projection data matrices";

  for (size_t energyWindowID=0; energyWindowID < m_energyWindowNb; energyWindowID++) {
  		for (size_t headID=0; headID < m_headNb; headID++) {
    		memset( m_data[energyWindowID][headID], 0, BytesPerProjection() );
   		}
  	}

    G4cout << " ... done "<<G4endl;
}

}


// Store a digi into a projection
// Modified by HDS : multiple energy window support
void GateProjectionSet::Fill( G4int energyWindowID, G4int headID, G4double x, G4double y)
{
  // Check that energyWindowID is valid
  if (energyWindowID<0)  {
    G4cerr << "[GateToProjectionSet::Fill]:" << G4endl
      	   << "Received a hit with a wrong energy window (" << energyWindowID << "): ignored!" << G4endl;
    return;
  }
  if ( static_cast<size_t>(energyWindowID) >= m_energyWindowNb) {
    G4cerr << "[GateToProjectionSet::Fill]:" << G4endl
      	   << "Received a hit with a wrong energy window (" << energyWindowID << "): ignored!" << G4endl;
    return;
  }


  // Check that the headID is valid
  if (headID<0)  {
    G4cerr << "[GateToProjectionSet::Fill]:" << G4endl
      	   << "Received a hit with a wrong head ID (" << headID << "): ignored!" << G4endl;
    return;
  }
  if ( (size_t)headID >=m_headNb) {
    G4cerr << "[GateToProjectionSet::Fill]:" << G4endl
      	   << "Received a hit with a wrong head ID (" << headID << "): ignored!" << G4endl;
    return;
  }

  // Check whether we're out-of-bounds
  G4int binX = static_cast<G4int>( floor ( ( x - m_matrixLowEdgeX ) / m_pixelSizeX ) );
  if  ( (binX<0) || (binX>=m_pixelNbX) ) {
    if (m_verboseLevel>=1)
      G4cerr << "[GateProjectionSet]: coordinate x (" << G4BestUnit(x,"Length") << ") outside the matrix boundaries ("
	     << G4BestUnit(m_matrixLowEdgeX,"Length") << "-" << G4BestUnit(-m_matrixLowEdgeX,"Length") << "): ignored!" << G4endl;
    return;
  }

  G4int binY = static_cast<G4int>( floor ( ( y - m_matrixLowEdgeY ) / m_pixelSizeY ) );
  if  ( (binY<0) || (binY>=m_pixelNbY) ) {
    if (m_verboseLevel>=1)
      G4cerr << "[GateProjectionSet]: coordinate y (" << G4BestUnit(y,"Length") << ") outside the matrix boundaries ("
	     << G4BestUnit(m_matrixLowEdgeY,"Length") << "-" << G4BestUnit(-m_matrixLowEdgeY,"Length") << "): ignored!" << G4endl;
    return;
  }

  // Increment the appropriate bin (provided that we've not reached the top)
  if (m_verboseLevel>=2)
      G4cout << "[GateProjectionSet]: binning hit at (" <<  G4BestUnit(x,"Length") << "," << G4BestUnit(y,"Length") << ") "
      << "into bin (" << binX << "," << binY << ") of head " << headID << G4endl;
  ProjectionDataType& dest = m_data[energyWindowID][headID][ binX + binY * m_pixelNbX];
  if (dest<USHRT_MAX)
    dest++;
  else
      G4cerr  << "[GateProjectionSet]: bin (" << binX << "," << binY << ") of energy window " << energyWindowID << "and head " << headID <<  " has reached its maximum value (" << USHRT_MAX
      	      << "): hit will be lost!" << G4endl;

  // Update the maximum-counter for this energy window and this head
  if (dest>m_dataMax[energyWindowID][headID]) {
    m_dataMax[energyWindowID][headID] = dest;
  }
}



/* Writes a head-projection onto an output stream

   dest:    	  the destination stream
   headID:    	  the head whose projection to stream-out
*/
void GateProjectionSet::StreamOut(std::ofstream& dest, size_t energyWindowID, size_t headID)
{
    dest.seekp(energyWindowID * BytesPerEnergyWindow() + headID * BytesPerHead() + m_currentProjectionID * BytesPerProjection(),std::ios::beg);
    if ( dest.bad() ) G4Exception( "GateProjectionSet::StreamOut", "StreamOut", FatalException, "Could not write a projection onto the disk (out of disk space?)!\n");
    dest.write((const char*)(m_data[energyWindowID][headID]),BytesPerProjection() );
    if ( dest.bad() ) G4Exception( "GateProjectionSet::StreamOut", "StreamOut", FatalException, "Could not write a projection onto the disk (out of disk space?)!\n");
    dest.flush();
}

/* PY Descourt 08/09/2009 */
void GateProjectionSet::FillARF( G4int headID, G4double x, G4double y, G4double ARFvalue)
{
  // Check that the headID is valid
  if (headID<0)  {
    G4cerr << "[GateToProjectionSet::FillFromARF]:" << G4endl
           << "Received a hit with a wrong head ID (" << headID << "): ignored!" << G4endl;
    return;
  }
  if ( (size_t)headID >= m_headNb) {
    G4cerr << "[GateToProjectionSet::FillFromARF]:" << G4endl
           << "Received a hit with a wrong head ID (" << headID << "): ignored!" << G4endl;
    return;
  }

m_rec++;

  // Check whether we're out-of-bounds
  G4int binX = static_cast<G4int>( floor ( ( x - m_matrixLowEdgeX ) / m_pixelSizeX ) );
  if  ( (binX<0) || (binX>=m_pixelNbX) ) {
    if (m_verboseLevel>=1)
      G4cerr << "[GateProjectionSet]: coordinate x (" << G4BestUnit(x,"Length") << ") outside the matrix boundaries ("
             << G4BestUnit(m_matrixLowEdgeX,"Length") << " , " << G4BestUnit(-m_matrixLowEdgeX,"Length") << "): ignored!" << G4endl;

   m_rej++;

    return;
  }

  G4int binY = static_cast<G4int>( floor ( ( y - m_matrixLowEdgeY ) / m_pixelSizeY ) );
  if  ( (binY<0) || (binY>=m_pixelNbY) ) {
    if (m_verboseLevel>=1)
      G4cerr << "[GateProjectionSet]: coordinate y (" << G4BestUnit(y,"Length") << ") outside the matrix boundaries ("
             << G4BestUnit(m_matrixLowEdgeY,"Length") << " , " << G4BestUnit(-m_matrixLowEdgeY,"Length") << "): ignored!" << G4endl;

   m_rej++;

    return;
  }


  // Increment the appropriate bin (provided that we've not reached the top)
  if (m_verboseLevel>=2)
  {
      G4cout << "[GateProjectionSet]: binning hit at (" <<  G4BestUnit(x,"Length") << "," << G4BestUnit(y,"Length") << ") "<< "into bin (" << binX << "," << binY << ") of head " << headID << G4endl;
      G4cout << " ARF Value " << ARFvalue << G4endl;

   }
  m_ARFdata[headID][ binX + binY * m_pixelNbX] += ARFvalue;



  G4double max =  m_ARFdata[headID][ binX + binY * m_pixelNbX] ;

  // Update the maximum-counter for this head
  if ( max - m_ARFdataMax[headID] > 0. ) m_ARFdataMax[headID] = max;


//G4cout << "  bin X " << binX<<"   bin Y " << binY << G4endl;

//G4cout << " m_ARFdata["<<headID<<"] = "<<m_ARFdata[headID][ binX + binY * m_pixelNbX]<<G4endl;

}
void GateProjectionSet::StreamOutARFProjection(std::ofstream& dest, size_t headID)
{
    dest.seekp(headID * ARFBytesPerHead() + m_currentProjectionID * ARFBytesPerProjection(),std::ios::beg);
    if ( dest.bad() ) G4Exception( "GateProjectionSet::StreamOutARFProjection", "StreamOutARFProjection", FatalException, "Could not write a projection onto the disk (out of disk space?)!\n");
    dest.write((const char*)(m_ARFdata[headID]),ARFBytesPerProjection() );
    if ( dest.bad() ) G4Exception( "GateProjectionSet::StreamOutARFProjection", "StreamOutARFProjection", FatalException, "Could not write a projection onto the disk (out of disk space?)!\n");
    dest.flush();
}
/* PY Descourt 08/09/2009 */
